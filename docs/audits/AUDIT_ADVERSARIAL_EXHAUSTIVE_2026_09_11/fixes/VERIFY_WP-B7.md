# VERIFY-WP-B7 — independent adversarial re-verification of WP-B7 (the asymptotic family: FGA routing, Y4 performance, Y5 structure, the uniform asymptotics, the GBD kernel clip, the JAX thin-screen displacement, the S6 fallback statistic, the pupil-chart sizing)

Branch `audit-fixes-2026-09`.  Subject: **f64444ec** (WP-B7).  Pre-change
baseline for every byte-identity and fail-before probe: **f64444ec^ =
c62c2f14**, which `git diff` confirms is byte-identical to the report's own
`b2baa505` baseline over all eleven files this package owns — so the report's
baseline choice stands and mine is the same library.

I did not write WP-B7.  Every number below was measured here, on fixtures the
report did not use, against oracles written for this verification.

**Headline.**  The package is sound.  Nine of the ten shipped claims reproduce,
several of them *more strongly* than the report states, and the byte-identity
claims hold archive-to-archive on 47 of my own 50 arrays — with the three that
differ being exactly the three the report says must move.  The two big
behaviour changes (items 10 and 11) are right, and item 11 recovers *more* on
my chart than on either of the report's: fidelity **0.000 → 0.895 / 0.989** at
twice the lens NA and **0.000 → 0.667 / 0.773** at four times it.

**But four things were wrong or unpinned, all found by re-deriving rather than
by reading, and all fixed here.**  The image-plane-waist cache keyed its
evaluator by `__qualname__` and **served one evaluator's width for another's**
(measured: `1.4163857683404920e-04` returned where the true answer was `None`);
it also ignores `_NEWTON_SCALE_RELATIVE_STOP`, the A/B seam this same package
added, which moves the width it measures — so an A/B measurement of that seam
through `aberration_tensor` was order-dependent.  `decompose_lg(only=)`
dropped a mode outside its `(p_max, ell_max)` rectangle in silence.  And
nothing pinned the `next_fast_len` half of S9: deleting it left all twenty
shipped ids green.

**And one stated mechanism is wrong.**  Item 9's residual is not "second order
(the OPL map is still the θ = 0 ray family's)".  It is **first order** and it
is the half of the re-referencing the fix did not do: the corrected screen
reproduces an exact conic raytrace *weighted the way the screen weights it* to
**0.00–0.04 %**, so everything that is left is the AMPLITUDE still being
sampled at the output pixel rather than at the entrance point.  That residual
has a closed form, `−B(A + dC)/d`, which is **zero at the image plane** and
grows linearly with the readout defocus (predicted +0.44 % / measured +0.45 %
on my optic; +0.10 % / +0.11 % on the report's own).

---

## 1. My oracles, and what they are worth

**Oracle V** (`scratchpad/vb7/oracle_v.py`) imports nothing from lumenairy:
explicit Schott Sellmeier dispersion, closed-form quadric intersection of a
conic of revolution (not Newton), vector Snell, ray-tube Jacobian from the
launch parametrisation, and a brute-force Kirchhoff / Rayleigh–Sommerfeld sum
from the exit surface with the two-sided obliquity.  Controls:

| oracle self-check | measured |
|---|---|
| my Sellmeier `n(N-LAK22, 1.064 µm)` vs the library's `get_glass_index` | 1.637409932 vs 1.637409932, **Δ = 0.0e+00** |
| my traced paraxial focal length vs the lensmaker equation | 4.9442 mm both |
| my traced best focus vs the paraxial BFD (`−A/C` of my own ABCD) | 4.6871 mm vs 4.6932 mm (the difference is this singlet's spherical aberration) |
| `AD − BC` of the ABCD I derive independently | 1.000000000000 |

**A NEGATIVE result worth recording, because it invalidated my first reading of
item 9.**  My first oracle launched rays 2 mm upstream at height `h` and
weighted them by `|E_in(h)|²`.  For a TILTED input that puts the ray at
`h + 2 mm·θ` on the entrance plane where `E_in` actually lives, displacing the
effective pupil centre by 47 µm and biasing the landing by −2.3 % — enough to
make the shipped correction look like a 2.3× over-correction.  Launching so
that the ray's height *at z = 0* is the weighting coordinate removes it
entirely and the numbers then agree with a closed-form paraxial prediction to
three digits.  A verification oracle that disagrees with the subject by a few
percent is far more often the oracle.

**Fixture C** (the asymptotic kernels) — unlike the report's stock N-BK7
f = 20/−20 mm order-6 M = 210 fit in every parameter: **N-SF11, R = +12 /
−35 mm, t = 1.6 mm, 7 mm aperture, object 250 mm, λ = 0.85 µm, order 5,
M = 126, 7⁴ chart rays, source box 15 µm, pupil box 0.012**, read on a
**decentred, non-square 27 × 33** output raster, with a decentred source point
and a decentred pupil centre.

**Fixture D** (the lens drivers) — **N-LAK22, R = +4.6 / −9.5 mm, t = 0.6 mm,
0.44 mm aperture, λ = 1.064 µm, N = 192–384, dx = 2.6 µm, w = 110 µm**,
f = 4.944 mm, NA 0.0469.

**Chart C** (the S6 ladder) — **N-SF6, R = ±9.0 mm, t = 0.9 mm, 0.90 mm
aperture, λ = 633 nm, N = 256, dx = 2.0 µm, w = 220 µm**, f = 5.761 mm,
NA 0.0822.  Different glass, wavelength, NA and pitch from both of the
report's charts.

---

## 2. Verdict table

| # | WP-B7 claim | verdict | my oracle and my numbers |
|---|---|---|---|
| 3 | FGA vs `phase_screen` measured; **no default moved** | **VERIFIED** (the no-move half, proved not read); the 15-setting convergence sweep **NOT re-derived** | §3.1 — `fga.py` is **AST-IDENTICAL** pre vs post with docstrings stripped, as is `lenses_gbd.py`; `_universal_route` returns the **same decision on all 30** (R, opd) cells of my own NA sweep, including the caustic-gate `fga` pick; **no signature default and no module constant moved** in any of the nine owned modules |
| 4a/b | Y4 fused basis + hoisted Newton factor, byte-identical | **VERIFIED** | §3.2 — 22 arrays + the modal field `np.array_equal` archive-to-archive on fixture C with a decentred raster, source point and pupil centre; and the two shipped `3e-8` pins read **identically pre and post** (9.615664e-09 / 9.615527e-09) |
| 4c | scale-relative Newton stop, opt-in, ships OFF | **VERIFIED** | §3.3 — the seam reads `False`; flipping it moves the `lg00` pin 9.615664e-09 → 9.652232e-09, i.e. **still 3.1× under the 3e-8 bar**, which is exactly the report's reason for making it a seam; `w_o` moves 6.3e-11 relative |
| 5 | `only=` + waist memo, byte-identical | **VERIFIED-WITH-NOTES — two key-completeness defects and one silent drop, all fixed here** | §3.4 / §4.1–4.3 — `L`, `w_o`, `sigma_grid_n` and the full `decompose_lg` rectangle byte-identical; all 12 named key arguments and all 14 fingerprint components miss correctly; but the `propagate` key was `__qualname__` (**false hit measured**), the Newton-stop seam is not in the key (**stale width measured**), and `only=` dropped an out-of-rectangle mode in silence |
| 6 | Y5 collapse cannot be made bit-identical as the copies stand | **VERIFIED, and the divergence is 4 decades larger on my fixture** | §3.5 — scalar vs batched on 5 chart points: `M` 0/5, `b` 0/5, `s1*` 0/5, `J` 0/5, `phi*` 1/5, `G0` 2/5, `detJ` 0/5 bit-equal; the two Newtons 0/5 with **\|Δv2\| = 2.71e-11** (1.06e-10 of the pupil half-range) against the report's 4.8e-15.  Identical pre and post, which is itself a byte-identity proof for the fusion |
| 7 | §15.9 premise corrected, table published, left as is | **NOT RE-DERIVED** (declared) | §6 — the dead-code correction is verified by reading `_lens_traced_uniform.py`; the f/1.92 fold table is a pure measurement with no code change and I spent the budget on the shipped behaviour instead |
| 8 | S9 GBD kernel clip, derived tolerance | **VERIFIED** (one unpinned half, pinned here) | §3.6 / §4.4 — **anamorphic diagonal Q at aspect ratios 1 … 1e6**: relL2 4.9e-16 … 1.3e-15 against the 9-sigma windowed sum and 5.8e-16 … 1.4e-15 against the true **dense** sum, per-axis clip working (Wx 10, Wy 10 → 95); a sub-pixel `R_cut` (0.0007 px) clamps to W = 1, never 0; `_fftconv_same` == `scipy.signal.fftconvolve(…, 'same')` to 3.7e-16 … 6.5e-16 for kernels from 1×1 to 255×255.  **Nothing pinned `next_fast_len`** |
| 9 | JAX chief-ray displacement | **VERIFIED — and the residual's stated mechanism is wrong** | §3.7 / §5.1 — exact conic raytrace on fixture D: landing error **−4.64 % → +0.45 %** at 0.25 / 0.5 / 1.0 × NA, off-axis converging **−11.76 % → −1.12 %** and **−57.30 % → −7.79 %**, diverging **+11.03 % → +0.78 %**; collimated byte-identical.  The term is **first-order complete** (the walk map is input-independent to 0.000 nm over an 11.234 µm walk; the omitted second-order term is ≤ 4e-3 waves).  The residual is the AMPLITUDE mis-indexing, in closed form |
| 10 | `_K1_DERIV_RESIDUAL_MAX = 1.2` | **VERIFIED-WITH-NOTES** | §3.8 — 19-input ladder on a **third chart**: the decision is right **two-sided** (nothing admitted-and-made-worse, nothing refused-and-made-better), but the last input where engaging still wins scores **8.99e-01** — **1.33× under the bar**, not the 2.1× the report's two charts implied.  Recorded in the source comment |
| 11 | pupil chart from mean + spread | **VERIFIED, and stronger than claimed; two comment corrections** | §3.9 / §5.2 — on my chart, **0.000/0.000 → 0.895/0.989** at 2× NA, **→ 0.879/0.979** at 3×, **→ 0.667/0.773** at 4×; s1 residual falls 34–55×; every previously-engaging case still engages; no runtime change above the noise, as the report says.  The comment's "the two rules differ by **2.3 %** at the bar" measures **2.83 %**, and the floor claim has an exception (§5.2) |
| B9-4 | `fga.py` comments corrected, no routing moved | **VERIFIED** | the AST-identity proof in §3.1 is stronger than the report's own statement |
| — | report §8.1 "**Not green when I finished**: 2 ids in the b1 file" | **SUPERSEDED** | the orchestrator applied the §8.1 patch inside f64444ec; that file is **33/33 green** at this commit.  The report's standing "not green" note should be struck when it is next touched (it is not mine to edit) |

---

## 3. The measurements

### 3.1 Item 3 and B9-4 — "no default moved", proved rather than read

I did not re-run the fifteen-setting FGA sweep (a pure measurement with no code
change, several hours).  What I *did* do is convert the report's weakest form
of claim — "I have **not** made that change" — into a proof.

**AST identity, docstrings stripped**, of all nine owned modules, PRE vs POST
(`ast.dump` after removing every module / class / function docstring):

| module | verdict |
|---|---|
| `lumenairy/propagators/fga.py` | **IDENTICAL** |
| `lumenairy/elements/lenses_gbd.py` | **IDENTICAL** |
| the other seven | changed (as they must be) |

So `fga.py`'s entire diff is docstrings and comments: the routing, the
whitelist and the dispatcher are bit-for-bit the pre-change code.  That is a
stronger statement than "no routing moved" and it cannot rot.

**Signature defaults and module constants**, PRE vs POST over the same nine
modules (218 → 226 functions with defaults, 61 → 66 constants): **no existing
default and no existing constant moved.**  The only differences are five new
opt-in keywords (`input_wavevector_saddle`, `scale_relative_stop`, `T12_rows`,
`only=` on two functions), all defaulting to `None`, and five new constants
(`_K1_DERIV_RESIDUAL_MAX`, `_NA_MEAN_MIN_FRACTION`, `_W_O_CACHE_MAX`,
`_NEWTON_SCALE_RELATIVE_STOP`, `_FFT_KERNEL_N_SIGMA`).  In particular
`_K1_FIT_RESIDUAL_MAX`, `_S1_FIT_RESIDUAL_MAX`, `_SADDLE_FLAT_INPUT_NA`,
`na_threshold = 0.12` and `_SIGMA_GRID_N_MAX_DEFAULT = 256` are untouched.

**`_universal_route`**, called with the dispatcher's own defaults on 6 radii ×
5 output distances of a 0.30 mm-aperture N-BK7 singlet at λ = 1.0 µm:
**0 of 30 decisions differ.**  The sweep reaches all three members
(`phase_screen` ×10, `traced` ×19, `fga` ×1 through the caustic gate), so it is
not a vacuous comparison.

### 3.2 Item 4a/b — byte identity on my own fixture

Archive-to-archive: `git archive f64444ec^ lumenairy` and
`git archive f64444ec lumenairy`, each extracted into its own scratch
directory, each imported from a **child process** whose cwd and `PYTHONPATH`
are that archive, with `lumenairy.__file__` asserted against it before any
other import.  Never through pytest, never against the shared working tree.
(The guard earned its keep: it caught a mis-typed root on the first run.)

50 arrays, on fixture C and fixture D.  **47 identical, 3 differ — and the
three are exactly the three the report says must move:**

| group | arrays | `np.array_equal` |
|---|---|---|
| Newton (`v2x*`, `v2y*`, `converged`), axial AND decentred source/pupil | 6 | ✔ 6/6 |
| `_compute_M_b_batch` (`M`, `b`, `s1*`, `J`, `phi*`, `G0`, `detJ`), axial AND decentred | 14 | ✔ 14/14 |
| `_phi_v2_hessian_batch`, both | 2 | ✔ 2/2 |
| `propagate_modal_asymptotic` field, 3 source × 3 pupil modes, decentred source point and pupil centre, 33 × 27 raster | 1 | ✔ |
| `decompose_lg` full rectangle (20 modes) — keys and values | 2 | ✔ |
| `aberration_tensor` `L` (5 × 2) and `(w_o, sigma_grid_n)` | 2 | ✔ |
| `apply_real_lens_maslov` on 14 centred inputs (collimated, converging, diverging, hard 0.60/0.80/0.95, speckle 0.05/0.30, Gaussian displaced ½ / 1 / 3 / 10 pixels, hard mask + 2-pixel displacement, off-axis converging) | 14 | ✔ 14/14 |
| the same driver, **complex64** input; `local_quadrature`; `fold_split=True` | 3 | ✔ 3/3 |
| `apply_real_lens_maslov_jax`, collimated / hard-apertured / 1-pixel-displaced (all real, non-negative) | 3 | ✔ 3/3 |
| `apply_real_lens_maslov` on a **tilted** input (item 11 must move it) | 1 | **differs**, relL2 1.87e-03 |
| `apply_real_lens_maslov_jax` on a **tilted** and on a **converging** input (item 9 must move them) | 2 | **differ**, relL2 3.41e-01 / 8.89e-02 |

The decentred / non-square raster and the decentred source point and pupil
centre are the arms that would catch a fusion that had quietly assumed an
axial, square, centred problem.  They do not.

**Measured a second way:** the two shipped `3e-8` bit-equality arms
(`TestAuditFixes…ModalAsymptoticStillBitEqual`) read
**max|new − cold_ref| / max(cold_peak, 1) = 9.615664e-09** (LG₀₀) and
**9.615527e-09** (4-mode) — *to every printed digit, identically, on the parent
and on WP-B7*.  A fusion that moved anything could not do that.

### 3.3 Item 4c — the opt-in Newton stop

`AM._NEWTON_SCALE_RELATIVE_STOP` is `False`.  Turning it on:

| quantity | stop OFF | stop ON |
|---|---|---|
| `lg00` bit-equality reading (bar 3e-8) | 9.615664e-09 | **9.652232e-09** (3.11× under the bar) |
| `test_w6_a2_v2_star`'s \|v2x*\| (floor 1e-15) | 1.907701e-16 | 1.907701e-16 (unmoved) |
| the image-plane waist `w_o` | 1.4163857683404920e-04 | 1.4163857683413807e-04 (6.3e-11 relative) |

So the report's stated reason for shipping it OFF — "9.1e-11 is under the 3e-8
the arms carry, so turning it on by default would have slipped past them" — is
correct and I can reproduce the slip: the reading moves by 0.38 % of itself and
the bar never notices.

**Pin headroom** (the brief asked for it):

| pin | reading | bar | headroom |
|---|---|---|---|
| `test_lg00_single_mode_bit_equal` | 9.615664e-09 | 3e-8 | **3.120×** |
| `test_lg_p0_4mode_prescription_bit_equal` | 9.615527e-09 | 3e-8 | **3.120×** |
| `test_w6_a2_v2_star_…` \|v2x*\| | 1.907701e-16 | 1e-15 | 5.24× |
| `test_w6_a2_v2_star_…` \|v2y*\| | **6.407877e-16** | 1e-15 | **1.56×** |

All four read identically on the parent, so WP-B7 moved none of them.  The
1.56× on `v2y*` is thin for a float64 round-off quantity across BLAS builds —
follow-up F3, not a WP-B7 regression.

### 3.4 Item 5 — the waist-cache key, attacked field by field

Every field the key names was mutated and must MISS.  All 26 do:

* the 12 named arguments — `s2x_img`, `s2y_img`, `source_point` x and y,
  `pupil_amplitudes` (set AND value-only), `w_s`, `w_p`, `v2_centre` x and y,
  `n`, and `propagate` with a different qualified name: **12/12 miss**;
* every component of `_fit_fingerprint` — `coef_phi`, `coef_s1x`, `coef_s1y`,
  the four centres, the four half-ranges, `wavelength`, `poly_order`,
  `extract_linear_phase`: **14/14 miss** at a 1e-9 relative nudge;
* the three `CanonicalPolyFit` fields the fingerprint does NOT name
  (`res_phi_rms_waves`, `res_s1_rms_m`, `n_rays`) do not move the answer, so
  omitting them is correct; a **reordered `multi_indices`** does move it and
  the fingerprint does separate it.

The key is therefore complete over what it names.  It is **not** complete over
what it does not — §4.1 and §4.2.

### 3.5 Item 6 (Y5) — the three copies, on a fixture the report did not use

Scalar `_compute_M_b` against `_compute_M_b_batch` at five chart points of
fixture C, each quantity scaled by its own global magnitude over the five
points (a per-point denominator is meaningless where `s1*` and `b` vanish by
symmetry at the chart centre):

| return | bit-equal | worst relative | report's reading |
|---|---|---|---|
| `M` | **0 / 5** | 1.161e-15 | 2/5, 1.1e-17 |
| `b` | **0 / 5** | 1.062e-15 | 3/5, 7.5e-18 |
| `s1*` | **0 / 5** | 5.460e-16 | 5/5, 0 |
| `J` | **0 / 5** | 5.966e-16 | 5/5, 0 |
| `phi*` | 1 / 5 | 1.386e-19 | 5/5, 0 |
| `G0` | 2 / 5 | 4.018e-78 | 5/5, 0 |
| `detJ` | **0 / 5** | 4.104e-16 | 5/5, 0 |
| scalar vs batched Newton `v2*` | **0 / 5** | **\|Δv2\| = 2.710e-11** on a 0.2560 half-range (1.06e-10 relative) | 0/5, 4.8e-15 (5.6e-14 relative) |

The report's conclusion — "a bit-identity harness across the three call sites
cannot be made green as they stand" — is therefore not merely right, it is
**understated**: on a different fit nothing at all is bit-equal and the Newton
divergence is four decades larger.  Every one of these readings is **identical
on the parent and on WP-B7**, which is an independent proof that the fusion
moved nothing.

`asymptotic_jax_twin.py` (the `xp` twin) is not in the commit's file list at
all, so `_compute_M_b_xp` is untouched by construction.

### 3.6 Item 8 (S9) — the clip, attacked anamorphically

The applicability gate refuses a skew `Q` but *admits a diagonal tensor `Q`
with unequal entries*, which is the case the per-axis clip exists for and the
one the report does not exercise.  I built uniform diagonal bundles whose y
decay is weaker than their x decay by factors 1 … 10⁶:

| aspect | Wx | Wy | applicable | relL2 vs the 9σ windowed sum | relL2 vs the **dense** sum | peak / grid |
|---|---|---|---|---|---|---|
| 1 | 27 | 27 | True | 1.345e-15 | 1.377e-15 | 9.74 |
| 4 | 27 | 54 | True | 7.708e-16 | 1.016e-15 | 11.87 |
| 25 | 27 | 127 | True | 6.864e-16 | 1.021e-15 | 17.84 |
| 10² | 27 | 127 | True | 6.519e-16 | 9.279e-16 | 17.84 |
| 10³ | 27 | 127 | True | 6.351e-16 | 7.956e-16 | 17.84 |
| 10⁴ | 27 | 127 | True | 6.686e-16 | 8.623e-16 | 17.84 |
| 10⁶ | 27 | 127 | True | 6.467e-16 | 8.022e-16 | 17.84 |

(N = 128, dx = 4 µm, z = 0.2 mm; the same sweep at N = 96, z = 0.05 mm reads
4.9e-16 … 8.2e-16 with Wx = 10 and Wy 10 → 95.)  The clip tracks each axis
independently, clamps at `N − 1` rather than wrapping, and never costs more
than the unclipped 37.7×.

Three more attacks on the same item:

* **a sub-pixel kernel.**  Scaling `Im(Q)` up so `R_cut` falls to 27.98 nm =
  **0.0007 px**, `_kernel_half_width` returns **1**, never 0, and the
  reconstruction still matches the dense sum to **3.234e-16**.
* **`_fftconv_same` against SciPy.**  `relL2` 3.745e-16 … 6.458e-16 against
  `scipy.signal.fftconvolve(a, G, 'same')` for kernels of 1×1, 3×3, 21×21,
  33×33, 41×41, 127×127 and 255×255, including kernels **larger than the
  array** — so the `next_fast_len` padding has not broken the `'same'` slice.
* **the unclipped regime.**  Against the 9σ windowed sum, N = 96 / 128 / 160 /
  192 at z = 2 and 8 mm: **7.99e-16 … 3.84e-15**, with `_fft_len` bumping
  3N − 2 → the next 5-smooth length each time (286→288, 382→384, 478→480,
  574→576).  Consistent with the report's 5.6e-16 … 8.3e-16 once the
  reference's own growth with N is allowed for.
* **the dark wings**, where a truncation would show first: at N = 128,
  z = 0.05 mm the max |error| / peak is 1.691e-15 in the brightest decade and
  **2.918e-16** at 10⁻⁶ of the peak.  The clip is invisible everywhere.

### 3.7 Item 9 — the JAX chief-ray displacement, on my optic

Screen, then `angular_spectrum_propagate` to one fixed plane 300 µm past the
traced best focus; intensity centroid in a window snapped to the exact trace's
own landing; oracle = my exact conic raytrace of the input's own rays,
**weighted by the input intensity at the entrance plane**.

| input | oracle landing | screen OFF (error) | screen ON (error) | improvement |
|---|---|---|---|---|
| collimated | 0.000 µm | 0.000 | 0.000 (**byte-identical**) | — |
| tilt 0.25 × NA | 61.388 µm | 58.538 (−4.64 %) | **61.667 (+0.45 %)** | **10.3×** |
| tilt 0.50 × NA | 122.799 | 117.099 (−4.64 %) | **123.362 (+0.46 %)** | 10.1× |
| tilt 1.00 × NA | 245.774 | 234.392 (−4.63 %) | **246.949 (+0.48 %)** | 9.6× |
| tilt 2.00 × NA | 492.953 | 466.364 (−5.39 %) | 483.192 (−1.98 %) | 2.7× |
| off-axis converging f = +40 mm, x₀ = 150 µm | −6.674 | −5.889 (−11.76 %) | **−6.599 (−1.12 %)** | 10.5× |
| off-axis converging f = +15 mm, x₀ = 150 µm | −3.792 | −1.619 (−57.30 %) | **−3.496 (−7.79 %)** | 7.4× |
| off-axis diverging f = −25 mm, x₀ = 150 µm | −11.173 | −12.406 (+11.03 %) | **−11.260 (+0.78 %)** | 14× |
| tilt 0.5 × NA **and** converging f = +40 mm | 122.807 | 117.110 (−4.64 %) | **123.362 (+0.45 %)** | 10.3× |

The f = +15 mm row's **−7.79 %** reproduces the report's own off-axis
converging **−8.03 %** on a different optic, wavelength and pitch — so that
number is real and not a fixture artefact.

**Is the correction first-order complete?  Yes, and here is the proof.**  The
walk `Δ = (xe − x, ye − y)` is a property of the element's map alone (the
Newton inversion never sees `E_in`), so it can be read straight out of the
shipped phase difference on a uniform tilt: `arg(on/off) = k₀ θ Δx`.  Over
three tilts spanning 4×:

* the recovered walk map is **input-independent to 0.000 nm over an 11.234 µm
  walk** — i.e. the applied term really is `k₀ k₁ · Δ` with a fixed geometric
  `Δ`, and it is odd in x with ±9.024 µm at the pupil edge;
* the y half works by symmetry;
* the term the correction **omits** for a quadratic input phase is exactly
  `−k₀|Δ|²/(2 f_in)`, which measures **3.7e-04 … 9.9e-04 waves rms** and at
  most 3.95e-03 waves peak at f_in = 15 mm — genuinely second order and
  genuinely negligible.

So the residual is **not** the omitted second-order walk.  §5.1 has what it is.

### 3.8 Item 10 — the slope bar on a third chart

The report derives `_K1_DERIV_RESIDUAL_MAX = 1.2` as the geometric mean of a
two-chart bracket 5.6e-01 … 2.7e+00.  I ran the same ladder — 19 inputs,
fidelity against the exact pointwise `'quadrature'` on the same chart, window
snapped to my exact trace's landing — on **chart C**, which neither of the
report's charts resembles:

| input | slope | value | ships | fid OPD-only | fid engaged | verdict |
|---|---|---|---|---|---|---|
| clean tilt 0.5 / 1.0 × NA | 9.25e-09 / 3.02e-08 | 5.5e-10 / 1.4e-09 | eng | 0.0000 | 0.9747 / 0.9697 | WINS |
| converging f = +40 mm | 4.28e-02 | 4.30e-02 | eng | 0.2029 | 0.9980 | WINS |
| diverging f = −25 mm | 4.44e-02 | 4.30e-02 | eng | 0.0685 | 0.9976 | WINS |
| speckle 0.002 / 0.010 on a tilt | 1.05e-02 / 5.27e-02 | | eng | 0.0000 | 0.9703 / 0.9446 | WINS |
| speckle 0.050 on a tilt | 2.77e-01 | 3.51e-02 | eng | 0.0000 | 0.1155 | WINS |
| **hard edge 0.80, CONVERGING carrier** | **8.99e-01** | 1.43e-01 | **eng** | **0.2028** | **0.9975** | **WINS (last)** |
| speckle 0.100 / 0.200 / 0.400 / 0.600 | 5.94e-01 / 1.31e+00 / 2.85e+00 / 4.31e+00 | | eng / OPD ×3 | 0.0000 | 0.0000 | tie |
| hard edge 0.80 / 0.60 / 0.40 on a tilt | 1.77e+00 / 4.63e+00 / 2.35e+00 | 6.5e-02 / 1.5e-01 / 4.6e-01 | OPD | 0.0000 | 0.0000 | tie |
| uniform white-noise phase on a tilt | 6.36e+00 | 8.97e-01 | OPD | 0.0000 | 0.0000 | tie |

**The decision is right, two-sided:** no input is admitted and made worse, and
none is refused and made better.  **But the margin below the bar is 1.33×, not
2.1×** — the last winner scores 8.99e-01 against a 1.2 bar, and it is a class
the report's ladder does not contain (a hard edge on a *converging* rather than
a tilted carrier, where engaging takes fidelity 0.203 → 0.998).  That is
narrower than TESTING_STANDARDS rule 5 wants.  Chart C cannot tighten the other
side: everything it refuses scores 0.000 on both arms there, so no loss is
measurable on it.  I have **not moved the bar** — the report set it and the
decision is still correct — but I have re-derived the comment beside it with
this reading, dated, so the next engineer sees the real margin.  Follow-up F2.

The report's own two-chart calibration also reproduces qualitatively: a clean
tilt scores 1e-08 or less, a converging/diverging input 4e-02, and a hard edge
on a tilted carrier 1.8–4.6 — two decades above any speckle row at the same
value residual, which is the mechanism the bar exists for.

### 3.9 Item 11 — the chart sizing, on my chart

Fixture D, readout 5.293 mm past the exit vertex, fidelity against the exact
pointwise `'quadrature'` on the same chart, window snapped to my exact trace's
landing, PRE archive vs POST archive in child processes:

| tilt / NA | `na_input` before → after | s1 fit residual before → after | engaged | `stationary_phase` fid | `local_quadrature` fid |
|---|---|---|---|---|---|
| 0.5 | 0.0707 → 0.0300 | 4.34e-05 → **1.01e-05** | T → T | 0.8732 → 0.8732 | 0.9860 → 0.9859 |
| 1.0 | 0.1583 → 0.0534 | 2.73e-04 → **2.46e-05** | T → T | 0.8803 → 0.8791 | 0.9872 → 0.9865 |
| 1.5 | 0.2322 → 0.0768 | 1.11e-03 → **5.24e-05** | T → T | 0.8900 → 0.8859 | 0.9885 → 0.9879 |
| **2.0** | 0.2815 → **0.1003** | 3.45e-03 → **1.02e-04** | **F → T** | **0.0000 → 0.8946** | **0.0000 → 0.9889** |
| **3.0** | 0.4571 → **0.1472** | 1.94e-02 → **3.15e-04** | **F → T** | **0.0000 → 0.8786** | **0.0000 → 0.9794** |
| **4.0** | 0.6076 → **0.1940** | 4.50e-02 → **8.13e-04** | **F → T** | **0.0000 → 0.6665** | **0.0000 → 0.7729** |

The recovery is larger than the report's own (its chart A reads 0.911/0.979 at
2× NA; mine reads 0.895/0.989 and keeps 0.879/0.979 at 3×, which the report
does not test).  The chart shrinks by 2.36–3.13×, the s1 residual falls by
34–55×, and **every case that engaged before still engages**.

**Runtime**: 0.026–0.050 s either way, dominated by process noise — I could not
measure a saving either, exactly as the report reports rather than claims.

**Byte identity for centred inputs** holds on all 14 of my own centred fields
(§3.2), including the two the brief asked me to attack: a Gaussian displaced by
½, 1, 3 and 10 pixels, and a pixel-quantised hard mask (at 0.60 / 0.80 / 0.95
of the pupil, and one displaced two pixels).  §5.2 has the boundary attack that
did find something.

---

## 4. Defects found and FIXED here

### 4.1 V7-1 (P2) — the waist cache served one evaluator's width for another's

`_measure_image_plane_waist`'s key ended in
`getattr(propagate, '__qualname__', repr(propagate))`.  That is not an
identity, and the report's own justification for it — "because a caller may
hand in a different evaluator" — is precisely the case it fails:

* two closures produced by the same factory share
  `factory.<locals>.propagate`;
* two lambdas share `<lambda>`;
* a `functools.partial` has **no** `__qualname__`, so the key degrades to a
  `repr` containing a **memory address** — the `id()`-reuse hazard
  `_fit_fingerprint` was written to avoid, reintroduced two lines later.

The library's own `lumenairy/_cache_registry.py::_clearer_identity` documents
this exact shape as a *measured* defect ("keying both on `type(fn).__name__` …
made ALL partials compare equal, so a partial of `clear_a` and a partial of
`clear_b` collided SILENTLY").

**MEASURED fail-before** (child process, the WP-B7 archive): two evaluators
built by one factory, one of them narrowing the probe field so its true answer
differs.

```
the two evaluators' __qualname__: 'make.<locals>.propagate' / 'make.<locals>.propagate'  equal: True
w_o(A) = 0.0001416385768340492; w_o(B) served = 0.0001416385768340492 (miss=False); w_o(B) computed = None
-> FALSE HIT: a different evaluator was served A's answer
```

The cache returned a finite width where the truth was "this probe cannot
produce one" — a different *verdict*, not a different last bit.

**The fix, and why it is the one it is.**  A code-object key
(`module, qualname, file, first line` — the registry's own shape) does **not**
close this: I implemented it first and the pin stayed red, because two closures
over one `def` share all four.  Refusing to cache anything but a plain
closure-free function closes it but breaks the shipped call-counting pin, whose
wrapper is necessarily a closure.  What is both exact and cheap is the
**object identity**, made sound by the cache entry itself: the key carries
`id(propagate)` and the entry stores `(value, propagate)`, so the callable
cannot be collected while its entry lives and no two live entries can share an
id.  The cost is 64 references at the cache bound; a caller that rebuilds an
equivalent wrapper every call simply never hits (correct, just uncached), while
the shipped caller — which passes the module-level
`propagate_modal_asymptotic` object — always does.

### 4.2 V7-2 (P3) — the A/B seam this package added is not in the key

`_NEWTON_SCALE_RELATIVE_STOP` changes what `propagate_modal_asymptotic`
returns, therefore what the probe measures, therefore `w_o`.  It was not in the
key, so an A/B measurement of the seam through `aberration_tensor` was
**order-dependent** — the exact measurement trap TESTING_STANDARDS is about.

**MEASURED fail-before:**

```
_NEWTON_SCALE_RELATIVE_STOP: w_o False 0.0001416385768340492 vs True 0.00014163857683413807 -> answer moves: True
with the cache warm, flipping the seam: miss=False, returned 0.0001416385768340492  -> STALE
```

Fixed: the key carries `_propagate_seams()`, a one-element tuple today, with a
comment saying that any future process-global the evaluator reads must join it
or drain the cache.

### 4.3 V7-3 (P3) — `decompose_lg(only=)` dropped an out-of-rectangle mode in silence

`_lg_mode_conj_stack` enumerates the `(p_max, ell_max)` rectangle and keeps the
members of `only`; a requested mode outside the rectangle is never enumerated,
so it vanished from the returned dict, and `aberration_tensor` — which fills
`L` from `overlaps.get(k_out, 0.0 + 0.0j)` — would have written a structural
zero into the tensor for a mode the caller asked for.  The docstring said the
modes "must lie inside the rectangle" and nothing enforced it.

No shipped call site can hit it (`aberration_tensor` derives `p_max`/`ell_max`
*from* `output_modes`), so this is latent, but silence is the wrong failure
mode for a library that refuses it everywhere else.  Fixed: `ValueError`,
CONVENTIONS §2 prefixed, naming the offending modes and both remedies.

### 4.4 V7-4 (P3) — the `next_fast_len` half of S9 was unpinned

Reverting `_fftconv_same`'s two `_fft_len(...)` calls to the naive
`Ny + Gy − 1` left **all twenty shipped b7 ids green**.  The report claims
1.45× on the transform from that padding alone at N = 512 and the changelog
lists `_fft_len` among the changed functions, so it is a claim with no gate.
Fixed by adding an integer-property pin (5-smooth, never shorter than the true
linear length, idempotent, a small bump above the awkward `3N − 2`, and
`_fftconv_same` counted calling it once per axis) plus a shift-and-add
convolution reference for the `'same'` slice.  It reds under the reversion.

### 4.5 Two numbers in shipped comments that do not reproduce

Not behaviour, but TESTING_STANDARDS is explicit that "right-conclusion,
wrong-numbers … reads as authoritative and it passes":

* `_NA_MEAN_MIN_FRACTION`'s comment (and report §7.4) says the two sizing rules
  "differ by **2.3 %** of a quantity that is already a 3-sigma margin" at the
  bar.  At the shipped bar `m = 0.1 σ₀` the difference is
  `(0.1 + 3√0.99)/3 − 1` = **2.832 %**.  (2.3 % corresponds to a bar of
  0.0785.)  Corrected in the comment, with the closed form.
* `_K1_DERIV_RESIDUAL_MAX`'s comment implies a ≥ 2.1× margin below the bar;
  §3.8 measures 1.33× on a third chart.  Recorded, dated.

---

## 5. Two stated mechanisms that are wrong, and what the right ones are

### 5.1 Item 9's residual is FIRST order, and it is the amplitude

The report says: "Corrected, it falls 14×; the residual −0.2 % is second order
(the OPL map is still the θ = 0 ray family's)."

It is not.  Write the element's paraxial entrance-plane → exit-vertex-plane ray
transfer as `[[A, B], [C, D]]` with `AD − BC = 1`.  At a distance `d` past the
exit vertex a uniform tilt `θ` lands at `θ(B + dD)`.  The uncorrected screen
adds the input's own `θ` to its collimated phase and lands at `θ d`.  The
correction's gradient is `k₀θ(dxe/dx − 1) = k₀θ(1/A − 1)`, so the corrected
screen lands at `θ d / A`.  Measured against that closed form:

| fixture | `A` | `1/A − 1` | OFF predicted / measured | ON predicted / measured |
|---|---|---|---|---|
| mine (R +4.6/−9.5, t 0.6, N-LAK22, 1.064 µm) | 0.949224 | +5.349 % | −4.661 % / **−4.65 %** | +0.439 % / **+0.45 %** |
| the report's own (R ±20.7, t 1.2, N-SF11, 1.55 µm) | 0.975281 | +2.535 % | −2.373 % / **−2.37 %** | +0.102 % / **+0.11 %** |

and, decisively, against the exact trace re-weighted the way the screen weights
its exit pupil (`|E_in|` at the EXIT coordinate rather than the entrance one),
over input waists from 0.06 to 0.39 of the aperture:

| waist / aperture | ON vs the true oracle | **ON vs oracle(exit-weighted)** |
|---|---|---|
| 0.068 / 0.125 / 0.250 / 0.386 (mine) | +0.45 / +0.45 / +0.45 / +0.42 % | **+0.01 / +0.00 / +0.00 / +0.04 %** |
| 0.060 / 0.130 / 0.250 / 0.400 (theirs) | +0.11 / +0.11 / +0.11 / −0.02 % | **+0.00 / +0.00 / −0.00 / −0.11 %** |

The corrected screen **is** the exact trace, to four decimal places, once the
pupil is weighted the way the screen weights it.  So everything left is the
AMPLITUDE: item 9 re-references the input's PHASE from the output pixel to the
entrance point and leaves `|E_in|` sampled at the output pixel.  In closed form
the residual landing error is

    d / (A (B + dD)) − 1  =  −B (A + dC) / (d + B(A + dC))  ≈  −B C Δz / d

with `Δz` the readout defocus — **exactly zero at the image plane** (where
`A + dC = 0` by definition of the back focal distance) and growing linearly
with defocus.  At my 300 µm defocus that predicts +0.446 % and measures
+0.45 %.

This is a *better* result for the fix than the report claims (it is exact, not
14× better), and it says precisely what the next increment is: resample the
amplitude at the entrance point too.  Follow-up F1.

### 5.2 `_NA_MEAN_MIN_FRACTION`'s floor has an exception, and it is the grid's Nyquist

The comment says the sub-bar readings are "round-off, not a launch direction"
and brackets the floor at 1.5e-02.  I widened the ladder to 41 fields on
fixture D.  The floor **reproduces**: the largest reading from a field with no
launch direction is **1.145e-02** (a uniform white-noise phase screen), with a
10-pixel-displaced Gaussian at 5.2e-07, a hard aperture at 0.60 of the pupil at
4.3e-04, 1.0 rad speckle at 6.3e-03, a one-pixel-wide Gaussian at 1.1e-03 and a
single live pixel at 9.0e-03 — and the smallest reading from a field that HAS
one is **4.168e-01** (a 1 mrad tilt).  So on a third fixture the bracket is
1.1e-02 … 4.2e-01 and the shipped 0.1 sits inside it, 8.7× above the floor and
4.2× below the first real direction.

**Two fields do cross the bar with no launch direction**, and both are the
grid's aliasing limit rather than a floor failure: a π-phase **checkerboard**
and π **stripes on x** — fields whose power sits AT the Nyquist frequency,
where `fftfreq`'s unpaired `−1/(2 dx)` column carries all of it — read
**5.536e-01**.  Their true mean launch direction is zero by symmetry; the
estimator cannot know that, for the same reason VERIFY-B1's item 7 found the
S6 aliasing blind spot "real but harmless".

What it costs is **bounded and one-signed**, which is the part worth writing
down: `m + 3√(σ₀² − m²)` exceeds `3σ₀` for every `m/σ₀ < 0.6` and peaks at
`√10/3 = 1.0541`, so a false positive can only make the chart up to **5.4 %
WIDER**, never narrower, while the false reading stays under 0.6 of the spread
— and a false *negative* just under the bar costs a chart 2.83 % narrow.  Both
bounds are now in the comment.

---

## 6. What I did NOT re-derive, and why

Declared rather than quietly skipped:

* **Item 3's fifteen-setting FGA convergence sweep and the five-point NA
  sweep** (report §2.2–2.3).  Pure measurement, no code change, several hours
  of brute-force Rayleigh–Sommerfeld.  I proved instead, decisively, that
  nothing shipped moved (§3.1), which is the part a future reader can be hurt
  by.  The escalation in report §8.3 is unaffected either way.
* **Item 7's f/1.92 fold table** (report §6.2).  Same reason.  I did verify by
  reading that `lumenairy/elements/_lens_traced_uniform.py` imports
  `_fold_airy_eval` and `pearcey`, so the "dead code" correction is right.
* **The wall-clock speed-ups** (1.86× / 2.41× / 1.68×, 12 203 → 7 439 ms,
  594.5 → 47.3 ms).  The b7 file deliberately pins operation counts instead,
  and I verified those counts and their fail-before (§7); timing a shared box
  against another engineer's concurrent run would produce a number worth less
  than the counts.

---

## 7. The pins: fail-before, and the mutation battery

**Fail-before on the parent.**  The b7 file against `f64444ec^`:
**19 failed, 1 passed**.  The one that passes is the WP-B9-derived aspheric
Jacobian pin, which is about behaviour WP-B9 already landed — correct.

**Mutation battery.**  Ten single-change reversions applied to an isolated
export of f64444ec, the b7 file run against each, one process at a time:

| mutation | red id |
|---|---|
| M1 drop the JAX displacement term | `…the_jax_screen_carries_the_chief_ray_displacement` |
| M2 make `_k1_fit_derivative_error` return 0 | `…the_k1_slope_error_sees_the_family_the_value_residual_cannot` |
| M3 revert the chart sizing to `3 σ₀` | `…a_uniform_tilt_sizes_the_chart_to_the_tilt_not_to_three_times_it` |
| M4 unclip the FFT kernel | `…the_fft_kernel_half_width_is_the_beamlets_own_support` |
| **M4b drop `next_fast_len`** | **none — 20 passed (defect V7-4, now pinned)** |
| M5 unfuse `_compute_M_b_batch` | `…the_batched_kernels_build_one_basis_per_sweep_not_three` |
| M6 un-hoist the Newton `s2` factor | `…the_batched_kernels_build_one_basis_per_sweep_not_three` |
| M7 ignore `only=` | `…decompose_lg_builds_only_the_modes_that_were_asked_for` |
| M8 disable the waist cache | `…the_image_plane_waist_probe_is_memoised_on_what_changes_it` |
| M9 flip the Newton stop default to True | `…the_scale_relative_newton_stop_is_opt_in_and_moves_the_iterate` |
| **M10 replace the fusion with a stacked `(3, M) @ (M, P)` GEMM** | `…the_fused_evaluation_reproduces_the_two_separate_ones_exactly` |

M10 is the one the report specifically says the `array_equal` assertion exists
to catch ("a GEMM is entitled to reorder the reduction against a GEMV").  It
does catch it.

**My four new pins**, fail-before on the unfixed build: 3 of 4 red immediately
(`…tells_two_evaluators_of_one_name_apart`,
`…key_carries_the_newton_stop_seam`, `…refuses_a_mode_outside_its_rectangle`);
the fourth (`…the_fft_transform_length_is_the_5_smooth_one`) reds under M4b,
which is its correct fail-before since the function it pins already exists.
All four red together under M4b: **4 failed, 20 passed**.

---

## 8. What I changed

| file | what |
|---|---|
| `lumenairy/propagators/asymptotic_aberration_tensor.py` | `_callable_identity` + `_propagate_seams`; the waist-cache key carries the evaluator's object identity and the Newton-stop seam; entries are `(value, fn)` so the id is sound; `_w_o_cache_put(key, value, fn)` |
| `lumenairy/propagators/asymptotic_modes.py` | `_lg_mode_conj_stack` refuses an `only=` mode outside the `(p_max, ell_max)` rectangle; `decompose_lg`'s docstring says so |
| `lumenairy/elements/lenses_maslov.py` | **comments only** — `_NA_MEAN_MIN_FRACTION`'s derivation re-measured (2.83 %, the third-chart ladder, the Nyquist exception and its one-signed bound) and `_K1_DERIV_RESIDUAL_MAX`'s margin re-measured on a third chart.  No code: its history fingerprints correctly still read OK, because they drop comments |
| `tests/unit/test_audit2609_b7_asymptotic.py` | **+4 ids** (section 8), all with a fail-before.  Nothing weakened, nothing removed |
| `docs/history/lumenairy.propagators.asymptotic_modes.md`, `…asymptotic_aberration_tensor.md` | re-recorded with `--reason` in this change |
| `docs/audits/…/fixes/VERIFY_WP-B7.md`, `VERIFY_WP-B7_CHANGELOG.md` | this report and its release text |

Nothing outside that list.  I did not touch `fga.py`, `gbd.py`,
`asymptotic_maslov.py`, `asymptotic_canonical_fit.py`, `_lens_jax.py` or
`lenses_gbd.py`, and **I moved no default the report did not move**.

---

## 9. Requested changes outside my ownership

1. **`docs/audits/…/fixes/WP-B7_REPORT.md` (WP-B7's own file).**  Two edits
   when it is next touched: (a) the standing "**Not green when I finished**:
   two ids in `test_audit2609_b1_maslov_input_wavevector.py`" is superseded —
   the orchestrator applied the §8.1 patch inside f64444ec and that file is
   33/33 green at this commit; (b) §7.2's "the residual −0.2 % is second order
   (the OPL map is still the θ = 0 ray family's)" should read "the residual is
   the input AMPLITUDE, still sampled at the output pixel; it is first order in
   the walk, vanishes at the image plane and grows linearly with the readout
   defocus (VERIFY-B7 §5.1)".  Also §7.4's 2.3 % → 2.83 %.
2. **Nothing else.**  Report §8.2 (the FGA whitelist widening) and §8.3 (the
   `_universal_route` caustic branch) are correctly escalated and I have not
   pre-empted them; both remain measured, specified and unapplied.

---

## 10. Follow-up

* **F1 (P2) — finish item 9's re-referencing.**  §5.1: the phase is
  re-referenced to the entrance point and the amplitude is not, and that is now
  the whole residual.  `amplitude='input'` should sample `|E_in|` at `(xe, ye)`
  by the same interpolation the OPL uses.  The gain is bounded and derivable:
  the landing error falls from `−BC Δz/d` to second order — +0.45 % → ~0 at my
  300 µm defocus, 0 → 0 at the image plane.  It is a behaviour change on the
  same keyword and belongs with whoever owns `_lens_jax.py` next.
* **F2 (P2) — `_K1_DERIV_RESIDUAL_MAX`'s lower margin is 1.33×.**  §3.8.  Not a
  defect today (the decision is right on all three charts), but the class that
  gets closest to the bar — a hard edge on a *converging* carrier — is not in
  the report's ladder.  A fourth chart would settle whether 1.2 wants to move
  up; I did not move it because the report set it and moving a bar on one new
  chart is how a bar gets un-derived.
* **F3 (P3) — `test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`'s floor.**
  §3.3: `|v2y*|` reads 6.407877e-16 against a 1e-15 bar — **1.56×**, on a
  quantity whose cross-build spread is exactly what TESTING_STANDARDS S4 is
  about.  Untouched by WP-B7 (identical on the parent), so not its regression,
  but the next asymptotic change will meet it.  The build-free restatement is
  "below the smallest pupil half-range the fit can produce, scaled", not an
  absolute 1e-15.
* **F4 (P3) — the `_NA_MEAN_MIN_FRACTION` gate's Nyquist reading.**  §5.2.  The
  consequence is bounded (≤ 5.4 % wider, never narrower) and it is the same
  aliasing blind spot the S6 estimator already has, so no change is warranted;
  it is recorded so a future widening of the gate does not assume a clean
  floor.
* **F5 (P3) — process-global seams and cross-call caches, generally.**  §4.2
  fixed the one seam `propagate_modal_asymptotic` reads today.  The library has
  several such seams (`_S6_INPUT_WAVEVECTOR_SADDLE`, `_QUAD_FACTORIZE`,
  `_NEWTON_SCALE_RELATIVE_STOP`) and several registry-cleared caches; a sweep
  asking "which cache's value depends on which seam" would be cheap and is not
  this package's job.
* **F6 (P3) — the waist cache's `id()` key and reload.**  §4.1's fix is exact
  for the process, but an `importlib.reload` of the propagators produces a new
  `propagate_modal_asymptotic` object and therefore a cold cache.  That is the
  correct answer (a reloaded module may be different code) and costs one probe;
  noted so it is not mistaken for a leak.

---

## 11. Tests run

All single-threaded (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`),
one process at a time, `-X faulthandler`, 2026-09-14.  Python 3.14.6,
NumPy 2.4.6, SciPy 1.17.1, JAX 0.10.1, Windows 11.

The shared working tree carries WP-B11b's in-flight edits, so **every run below
is in an isolated export**: `git archive f64444ec` into a scratch tree, with
only my four changed files and two history documents copied over it.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b7_asymptotic.py` (fixed tree) | **24 passed** (20 shipped + my 4) | 23.1 s |
| `pytest …b7_asymptotic.py …w3_oracles.py …w4_p5_return_contract.py` | **258 passed** | 79.6 s |
| `pytest …a4_maslov_gbd …a4_verify_maslov_asymptotic …a4_asymptotic …a4_fga_s10 …r_guards_and_merits` | **129 passed, 2 skipped** | 123.9 s |
| `pytest …w6_asymptotic …perf_v4_12_0_asymptotic …v5_21_gbd_maslov_perf …niche_k4_uniform_caustic …niche_r2_pearcey_cusp` | **95 passed** | 233.9 s |
| `pytest …b1_maslov_input_wavevector.py …::…ModalAsymptoticStillBitEqual` | **35 passed** (the b1 file is 33/33 at this commit) | 39.7 s |
| `pytest …a17_history_lint.py …a17_history_relocation.py` | **744 passed** | 42.5 s |
| `pytest tests/unit -k "maslov or asymptotic or gbd or fga"` | **791 passed, 8 skipped, 0 failed**, 15318 deselected | **43 min 3 s** |
| `python validation/run_all.py test_lenses test_propagation` | **ALL 2 files passed** (25.6 s / 7.0 s) | 32.8 s |
| `ruff check` on my four changed files | **All checks passed** | 1 s |
| `scripts/record_history_fingerprints.py <module> --check` × 3 | **OK** on all three (`lenses_maslov` unchanged: comments only) | 6 s |
| baseline: b7 file against the unmodified f64444ec export | **20 passed** | 24.0 s |
| fail-before: b7 file against the `f64444ec^` export | **19 failed, 1 passed** | 9.7 s |
| mutation battery, 10 reversions × the b7 file | table in §7 | 1 min 33 s + 22 s |
| M4b + my new pin | **4 failed, 20 passed** | 21.0 s |

The broad selection ran to completion with no access violation; the DENSE
`reconstruct_field_from_beamlets` path the landing warns about
(`test_fft_reconstruct_anamorphic_diagonal_Q`) passed in both the 95-test batch
and the 43-minute selection.

Probe runs (each in its own read-only archive, child process,
`lumenairy.__file__` asserted):

| probe | what | duration |
|---|---|---|
| `p1_bytes.py` pre / post + `cmp_bytes.py` | 50 arrays, 47 identical (§3.2) | 17 s / 19 s |
| `p2_namean.py` | the 41-field `_NA_MEAN_MIN_FRACTION` ladder and the jump at the bar (§5.2) | 9 s |
| `p3_gbd.py` / `p3b_gbd.py` | S9: anamorphic 1 … 1e6, unclipped regime, SciPy cross-check, dark wings, sub-pixel kernel (§3.6) | 47 min / 10 min |
| `p4_jaxwalk.py`, `p4b_slope.py`, `p4c_waist.py` | item 9: the landing table, the walk map, the waist sweep that isolated the amplitude (§3.7, §5.1) | 1–3 min each |
| `p5_k1slope.py` | item 10's 19-input ladder on chart C (§3.8) | 1 min 58 s |
| `p6_waistcache.py` | the 26-field key attack and the two false-hit demonstrations (§3.4, §4.1–4.2) | 1 min |
| `p7_defaults.py` pre / post | AST / signature / constant / routing identity (§3.1) | 6 s each |
| `p9_headroom.py` full / parent | the two `3e-8` arms and the 1e-15 floor (§3.3) | 9 s each |
| `p10_item11.py` pre / post | item 11's recovery table (§3.9) | 2 min each |
| `p11_y5.py` pre / post | the Y5 divergence (§3.5) | 20 s each |

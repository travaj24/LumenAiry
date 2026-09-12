# WP-A6 -- the traced-carrier chain (`propagators/carrier.py`, `carrier_field.py`)

Findings C1-C5 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.4, derivations in
`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/CARRIER.md`.  Branch `audit-fixes-2026-09`,
no git write commands issued.  Every number below is a measurement made on this
machine with `OPENBLAS_NUM_THREADS=1`, before and after, against the audit's own
repro fixtures or an independent oracle written for the purpose.

---

## 1. Summary

| # | Status | Files : lines | Tests | Oracle | Measured before -> after |
|---|---|---|---|---|---|
| **C1** | **fixed** (resolver + guard) | `carrier.py:3125` `_beam_gaussian_radius`, `:3140` `_beam_containment_standoff`, `:3225` `_check_focus_containment`, `:3032` `_default_focus_standoff`, `:2694` `carrier_referenced_focus_readout`, `:381` `_FOCUS_READOUT_CONTAINMENT_MIN`, `:3321` + `:9425/9467` stage publication | `test_audit2609_a6_carrier.py::TestC1FocusReadoutContainment` (11 tests); `test_niche_r9_highna_final_leg.py::test_r9_exact_leg_focuses_highna_sphere` (strengthened) | the SAME physical field read against its own carrier (itself scored by the audit against an analytic Gaussian-ABCD focal oracle at relL2 1.05e-03 / peak 0.99992), plus an analytic waist oracle | focal peak vs truth **0.986 / 0.745 / 0.188 / 0.026 -> 0.99951 / 0.99860 / 0.99430 / 0.98524** at R/R0 = 0.99 / 0.98 / 0.95 / 0.90; stop-plane containment **1.96 / 1.39 / 0.91 / 0.87 -> 3.20** (= `_FOCUS_STANDOFF_MARGIN`); warnings 0 -> 0 |
| **C2** | **fixed** | `carrier.py:2084` `_fit_carrier_inv` (`centre=`, tilt projection), `:2254` `carrier_referenced_fit_radius` (`centre='auto'`) | `...::TestC2FitRadiusCentre` (18 tests) | the radius / the tilt known BY CONSTRUCTION of the field | decentred parabola `R_fit/R` **1.5000 / 3.0000 / 9.0000 -> 1.0000** at 0.5 / 1 / 2 waists; decentred pure tilt `R_fit` **0.0750 / 0.0075 / 0.0113 m -> 4.8e15 / 2.5e14 / 6.8e13 m** (truth `inf`) |
| **C3** | **fixed** | `carrier.py:1343` `_carrier_step_fast`, `:8846`, `:8992`, `:9462`, `:9486` (obliquity pistons); `carrier_field.py:434` `phasor_on(dtype=)`, `:783` `full_field`, `:849` `from_full_field`, `:1435` `re_reference`, `:1656` `aggregate` | `...::TestC3Complex64` (5 tests) | numpy's own NEP 50 promotion rules; the returned `dtype` | tilted complex64 chain output **complex128 -> complex64**; `CarrierField.full_field()` **complex128 -> complex64**; complex64 phasor within 1.2e-07 of the narrowed complex128 one |
| **C4** | **fixed** | `carrier.py:870` `_SEPARABLE_CARRIER_PHASE`, `:873` `_radial_carrier_phase`, `:6078` `_tilt_ramp`, `:2415` `_rereference`, `:1163` `_exact_envelope_tf_step`, `:543` `_tf_phase_to_H`, `:1600` `_asm_axis`, `:4018` `_fourier_upsample_crop`, `:1357` `_envelope_amp_radius` | `...::TestC4SeparablePhases` (18), `...::TestC4TransferFunction` (13) | whole-grid `meshgrid` + `np.exp` expressions written in the test; raw-`numpy` transforms; `tracemalloc` | `_radial_carrier_phase` **197.0 -> 28.6 ms / 3.50 -> 1.01 grids** (N = 2048), **805.5 -> 70.1 ms** (N = 4096), diff 1.1e-13 / 5.7e-13; `_tilt_ramp` **4.9x**; `_exact_envelope_tf_step` **4.00 -> 2.00 grids at exactly 0.0 difference**; `reconstruct` **248.6 -> 55.8 ms / 3.50 -> 2.00 grids** (N = 2048) |
| **C5** | **fixed** (7 items) | `carrier.py:470` (dead `_freq_sq_1d` deleted), `:1600` `_asm_axis` mask, `:9410` + `:9450` gap-kernel / tilt forwarding, `:3848` `_sphere_parab_conversion(dy=)`, `carrier_field.py:652` `CarrierField` freeze cycle, comment corrections at `carrier.py:1284`/`1339` | `...::TestC5SmallerItems` (15, incl. the JAX backend-parity check); `test_carrier_field.py::test_mutating_a_built_carrier_field_is_deprecated`; `test_niche_c5_...` (re-barred) | `np.fft.fftfreq`; a hand-built 1-D angular spectrum; a hand-built eikonal | `_freq_sq_1d` **present (wrong at odd N) -> deleted**; `_asm_axis` at N = 65/127 **half a bin out of register -> relL2 0.000e+00 vs the oracle**; `dy=3dx` conversion **0 -> exact match to the oracle** |
| §15.9 | **deferred, design below** | -- | -- | -- | -- |

Nothing the audit listed under *checked and found correct* moved: re-measured
below (§2.6).

---

## 2. Per finding

### C1 (P1) -- the focus readout sized its stop grid from the CARRIER, not the beam

**What was wrong.**  `_default_focus_standoff` and `_near_focus_needs_bridge`
both estimate the beam at the stop plane from the carrier:
`w0 = lambda|R|/(pi w_env)`, waist at `-R`, `w(s) = w0 sqrt(1+(s/zR)^2)`.  The
co-moving grid contracts by `|R_out/R|` by construction, so the containment
margin `ext f/sqrt(1+f^2)` the whole `_FOCUS_STANDOFF_MARGIN` derivation is
written in is a statement about the CARRIER's geometric contraction.  It is only
a statement about the beam when the envelope is flat.  The chain takes its
carrier from a paraxial ABCD (`_paraxial_group_r_out`), and the real exit
wavefront differs from that by the Gaussian `zR^2/delta` term, by aberration and
by the `~NA^2/2` sphere-vs-parabola offset the module documents -- so the
mismatch is structural.  Nothing between the carrier leg and the Bluestein zoom
measured what actually landed; the replica guard is a window-vs-period test and
cannot see it.

**What I changed, and why in two independent pieces.**

1. *The resolver measures the beam* (`_beam_containment_standoff`, `carrier.py:3140`).
   The envelope's residual inverse-curvature is fitted with the module's own
   `_fit_carrier_inv` (`estimator='increment'`, centred on the amplitude
   centroid -- the C2 fix is what makes that reading trustworthy) and composed
   with the carrier, `1/R_eff = 1/R + 1/R_env`.  Writing the containment
   condition `half*|zeta - zeta_cf|/zeta_cf >= M * w_beam(zeta)` with the beam's
   Gaussian ABCD width gives a QUADRATIC in the stop-plane position whose value
   at the carrier focus is negative, so the carrier focus always lies between
   the roots and the containment region on the input side is `zeta <= zeta_-`.
   Closed form, no scan, no iteration.
2. *The result is measured* (`_check_focus_containment`, `carrier.py:3225`,
   knob `on_focus_containment`, default `'error'`).  A model is a model: the
   guard reads the beam that landed.

They are independent because a model that is wrong in a way the resolver cannot
anticipate (a strongly aberrated or multi-lobed envelope) still has to be caught,
and because the guard alone would only report the failure while the resolver
alone would leave no backstop.

**Why the resolver can only lengthen, and why that keeps the shipped universe
intact.**  `_beam_containment_standoff` returns exactly `0.0` when the fitted
residual is exactly `0.0` -- which is exact for a real/flat envelope -- and the
caller takes `max(s_shipped, s_beam)`.  Measured over the resolver's own
calibration matrix (6 NA x 10 grid extents = 60 cells, flat envelopes): **0 cells
with a non-zero beam term**, and `_default_focus_standoff` bit-identical to the
pre-fix value in all 60.  The two models differ only in evaluating the
diffraction term at the stop plane instead of at the carrier focus, so the beam
term is never LONGER than the shipped law at `R_eff == R` (measured 1.1 % / 0.5 %
/ 0.2 % shorter on three cells) -- which is what makes the resolved standoff
continuous in the envelope's residual curvature rather than stepping at the
short-circuit.

**Why the guard's floor is 1.0 and not the margin the resolver targets.**  On a
grid narrower than `M/sat = 3.695` beam radii the resolver *deliberately*
undershoots the margin (defects V1/V2), down to a measured containment of 1.257
at the narrowest cell of its own matrix.  A bar at the target would refuse the
module's own documented small-extent branch.  The bar is therefore set where the
beam stops FITTING: at containment 1.0 the half-width is one amplitude radius,
~25 % of a Gaussian's power lies outside the outer quarter of the grid, and the
transport is an FFT, so that power wraps rather than vanishing.

**Verification (numbers).**  `repro/CARRIER/p6c_mismatch.py`, shipped defaults,
one physical field, only the reference carrier varied:

```
R/R0   standoff        containment        peak vs truth      warnings
        before -> after  before -> after   before -> after
1.00    222.4 ->  222.4   3.21 -> 3.21     1.000000 -> 1.000000    0 -> 0
0.99    418.0 -> 1008.6   1.96 -> 3.20     0.986188 -> 0.999514    0 -> 0
0.98    613.6 -> 1874.0   1.39 -> 3.20     0.745432 -> 0.998602    0 -> 0
0.95   1200.7 -> 4173.3   0.91 -> 3.20     0.187913 -> 0.994304    0 -> 0
0.90   2180.1 -> 7144.9   0.87 -> 3.20     0.026309 -> 0.985236    0 -> 0
```

`repro/CARRIER/p6_invariance.py` (the same question asked as an invariance):
relL2 against the matched-carrier reference is now 5.1e-04 / 7.6e-03 / 3.2e-02 at
R = -23 / -19.6 / -18 mm with peak ratios 0.999898 / 0.998602 / 0.985236.

The guard, fed the PRE-FIX leg (recovered by reverting the resolver's beam term
in process -- its only change), refuses exactly the rows that had collapsed:

```
R/R0  pre-fix leg  containment (meas / model)  peak/truth   default refuses
0.99     418.0 um      1.963 / 1.963            0.986188        no
0.98     613.6 um      1.387 / 1.374            0.745432        no
0.95    1200.7 um      0.909 / 0.697            0.187913       YES
0.90    2180.1 um      0.863 / 0.365            0.026309       YES
```

and is silent on all 60 cells of the flat-envelope matrix (worst containment
1.257, worst `window_energy_frac` 0.999863).  The matched readout still
reproduces the analytic waist: relL2 4.728e-05, measured radius / `w0` = 1.000016.

**Residual risk.**  (a) The R/R0 = 0.98 row (containment 1.387, peak 0.745
pre-fix) sits ABOVE the 1.0 floor, so the guard alone would not have caught it --
the resolver is the primary remedy and the guard is the backstop, not the
reverse.  The two-sided clearance around the floor is ~1.2x (0.909 below, 1.257
above), not decades; that is stated in the code and in the test, and it is why
the floor refuses the unfittable rather than grading quality.  (b) The model half
of the guard assumes a Gaussian; on a genuinely non-Gaussian envelope it can read
pessimistically -- measured, it does exactly that on the high-NA fail-before arm
of `test_niche_r9_highna_final_leg` (0.419 model vs 1.055 measured), where the
refusal is nevertheless CORRECT (that arm is the one the test asserts
`ee_par < 0.10` on).  (c) Cost: see §2.5.

### C2 (P1) -- `carrier_referenced_fit_radius` fitted about the grid origin

**What was wrong.**  `_fit_carrier_inv` built `x`/`y` about the grid origin with
no `centre` argument.  The estimator is a moment, so a beam centred at `x0`
carrying a perfect parabola about its own centre loses the `x0 <x>` cross term
and reads `1/R_fit = (1/R) 2 sigma^2/(x0^2 + 2 sigma^2)`, i.e.
`R_fit = R(1 + 2x0^2/w^2)`; and a FLAT wavefront carrying a uniform tilt `L`
reads `1/R_fit = L x0/(x0^2 + 2 sigma^2)` -- a finite radius where the truth is
`inf`.  `on_aliased` cannot fire (the field is perfectly sampled).

**What I changed.**  `centre` on `_fit_carrier_inv` (default `(0.0, 0.0)`,
short-circuited) and `centre='auto' | 'origin' | (x0, y0)` on
`carrier_referenced_fit_radius`, defaulting to `'auto'` = the intensity centroid.
A non-default centre also PROJECTS OUT the residual tilt: the `w`-weighted mean
phase slope times `sum(w*x)` is subtracted from the moment, which makes the fit
tilt-immune even when the centre is not exactly the centroid.  The projection is
written once (`_tilt_free_moment`) and applied identically to both estimators;
the default branch keeps the historical ASSOCIATION `(w*x)*slope`, not merely the
same operands, because re-grouping it moves the answer by a few ulp and a
deterministic-fit pin would see that.

**Verification.**  `repro/CARRIER/p4_fit.py`, section 3 (decentred parabola,
truth 1.0000):

```
            gradient            increment
x0          before -> after     before -> after
0.5 w       1.5004 -> 1.0002    1.5000 -> 1.0000
1.0 w       3.0007 -> 1.0002    3.0000 -> 1.0000
2.0 w       9.0022 -> 1.0002    9.0000 -> 1.0000
```

The residual 1.0002 on `'gradient'` is the documented amplitude-curvature bias
`0.5 (dx/w)^2 = 0.5*(2/100)^2 = 2.0e-4` of the central difference -- a GRID
artefact, not a centring one, and the same value on axis.  Section 2 (decentred
pure tilt, truth `inf`): the smallest post-fix |R| across the six cases is
1.18e8 m against 0.0075 m pre-fix.  Sections 1, 4 and 5 (on-axis fits, the
`R -> inf` guards, the aperture round trip) are unchanged to every printed digit.

Byte-identity on axis is asserted directly (`centre='auto'` vs `'origin'` over
four radii, both estimators, scalar and astigmatic).

**Residual risk.**  `carrier_referenced_aperture(refit_carrier=True)` keeps its
internal fit about the grid ORIGIN.  That is deliberate and now documented: it
hands the radius to `_rereference`, which builds `exp(i k r^2/2 (1/R_old -
1/R_new))` on the centred grid, so a fit about any other point would hand it a
radius its own screen cannot express and the residual would be re-interpreted as
envelope content.  A decentred caller should fit with `centre='auto'` and pass
the result as `new_carrier=`.  Making that pair decentre-aware end to end is a
larger change (it needs a decentred `_rereference`) and is not in this WP.

### C3 (P2) -- complex64 chains promoted to complex128

**What was wrong.**  Four chain sites and `_carrier_step_fast`'s NumPy amplitude
scale multiplied by a numpy complex128 SCALAR, which is strong under NEP 50.
`_carrier_step_fast` knew about the hazard and worked around it on the backend
branch only.  In `carrier_field.py`, `CarrierSpec.phasor_on` built the whole
reference phasor in complex128 unconditionally even though `_tilt_ramp` and
`_tilt_exactness_phase` had grown `dtype=` in v5.44 for exactly this, and
`aggregate` hard-coded a complex128 accumulator.

**What I changed.**  `complex(...)` at all five scalar sites; `dtype=` on
`phasor_on` routing the sphere through `_phasor_rows` and forwarding to the two
helpers; `full_field` / `from_full_field` / `re_reference` pass the envelope's
own dtype; `aggregate`'s accumulator sized from
`np.result_type(np.complex64, *[f.envelope.dtype for f in fields])`, the rule the
multi orchestrator already uses.

**Verification.**  `repro/CARRIER/p8_c64chain.py`: a one-singlet chain with a
complex64 input and `TiltedCarrier(inf, L=0.02, ...)` returns **complex128 ->
complex64** (the untilted control was and stays complex64).
`repro/CARRIER/p7_odd_dtype.py`: `CarrierField.full_field()` **complex128 ->
complex64**.  The complex64 phasor is the narrowed complex128 one to one float32
rounding (measured max |diff| <= 1.2e-07 with a piston of 3 mm, i.e. ~1.4e4 rad
of argument -- the boundary `_phasor_rows` documents).  `phasor_on`'s own default
stays complex128.

**Residual risk.**  The chain's TILTED paraxial landing still returns complex128,
because the chief-ray ramp it applies to the readout field
(`exp(i k0 (L u + M v))` on the `N_out^2` output grid) is a genuine complex128
ARRAY.  That grid is small (the readout window, not the chain grid), so the
memory consequence is negligible; making it dtype-aware would need
`angular_spectrum_propagate_mft` to preserve complex64, which is not my file.
Listed in §5.

### C4 (P2, perf) -- separable phases, the transfer-function build, the fine-grid rescale

Measured with `time.perf_counter` medians of >= 5 interleaved runs and
`tracemalloc` peaks; no test asserts a timing (TESTING_STANDARDS S1).  What the
tests assert is the pair of properties that make each change safe: the agreement
bound (or bit-identity), and the ALLOCATION count, which is a property of the
code and not of the machine.

| change | time | peak (complex128 full grids) | difference |
|---|---|---|---|
| `_radial_carrier_phase` separable, N = 2048 | 197.0 -> 28.6 ms (**6.9x**) | 3.50 -> 1.01 | 1.139e-13 |
| ... N = 4096 | 805.5 -> 70.1 ms (**11.5x**) | 3.50 -> 1.00 | 5.684e-13 |
| `_tilt_ramp` separable, N = 2048 | 146.8 -> 30.1 ms (**4.9x**) | -- | 8.544e-14 |
| `_rereference` separable, N = 1024 | -- | 3.50 -> 2.01 | 2.5e-14 rel |
| `_exact_envelope_tf_step`, N = 2048 | 411.9 -> 267.5 ms (**1.54x**) | 4.00 -> 2.00 | **exactly 0.0** |
| `carrier_referenced_reconstruct`, N = 2048 | 248.6 -> 55.8 ms (**4.5x**) | 3.50 -> 2.00 | 2.3e-15 |
| `_envelope_amp_radius`, N = 2048 | 98.2 -> 60.8 ms (**1.6x**) | -- | **bit-identical** |
| `_fourier_upsample_crop` rescale | -- | one fine grid saved (4.29 GB at `n_fine_cap = 16384`) | 5.3e-16 rel vs a raw-numpy oracle |

The separable regrouping is exact in exact arithmetic; in float64 the measured
1.1e-13 / 5.7e-13 sits two decades under the ~1e-11 rad representation noise of
the `k r^2/2R` ~ 1e5-1e6 rad arguments these screens carry, i.e. inside the
existing noise rather than being a new approximation.
`_SEPARABLE_CARRIER_PHASE = False` restores the whole-grid build bit for bit and
is exercised as the fail-before switch.

The transfer-function change is **bit-identical at every shape tested**
(N = 63/64/65/128/256, tilted and untilted, `np.array_equal` on the raw bytes)
because `np.exp` of a pure-imaginary argument IS `cos + i sin` through the same
libm, and the untilted short-circuit re-associates by addition and multiplication
only -- both commutative to the bit in IEEE-754.  That is better than the audit's
own estimate, which expected 7.3e-12 for the untilted fast path; the difference
is that `kx^2[None,:] + ky^2[:,None]` is the same sum of the same two operands as
`ax*ax + ay*ay` at zero tilt, so no re-association of the RADICAND was needed.

### C5 (P3) -- the smaller items

* **`_freq_sq_1d` deleted** (`carrier.py`).  Zero call sites across `lumenairy/`
  (`grep -rn '_freq_sq_1d\b'` finds only the definition and the false
  cross-reference), and it still carried the `- N/2` offset D7 fixed everywhere
  else -- measured at N = 5: `[9.87 3.55 0.39 0.39 3.55]` against `_bld`'s
  `[6.32 1.58 0. 1.58 6.32]`.  The `_freq_sq_1d_bld` docstring's claim that they
  agreed is gone with it; the surviving builders equal `fftfreq` to <= 1 ulp at
  both parities (exactly, wherever `1/(N d)` is representable -- the residual is
  that `fftfreq` multiplies by the reciprocal where these divide).
  This makes `repro/CARRIER/p7_odd_dtype.py`'s first section un-runnable by
  construction: it imports the deleted name.  The other two sections were
  re-measured from a scratch copy with that import removed.
* **`_asm_axis` band-limit mask** now uses the `N//2` offset its own transfer
  function is built on.  Measured relL2 against a hand-built 1-D band-limited
  angular spectrum: **0.000e+00** at N = 64, 65 and 127.  Even `N` is unchanged
  (`N/2 == N//2` exactly).
* **`gap_kernel` forwarded** to the paraxial readout's internal carrier leg
  (`_par_kw.setdefault('gap_kernel', gap_kernel)`), and the chain's `gap_kernel`
  documented for the first time (it had no Parameters entry).  **Not
  reproducible as stated for the fine retrace**: the audit lists
  `_fine_trace_group_exit` (`carrier.py:8342` at HEAD) as a site the kernel
  fails to reach, but that function runs no free-space transport at all -- it is
  a band-limited re-grid plus a ray trace (`grep` over its 516 lines finds no
  `propagate_*`, `angular_spectrum*`, `_envelope_tf_step` or `fresnel`).  Same
  for `carrier_referenced_exact_focus_readout`, which is an exact Bluestein
  angular spectrum.  There is no kernel there to select; that is now stated in
  the chain's docstring instead of forwarded.
* **`carrier_referenced_focus_readout` gains `tilt=`** and the chain forwards the
  congruence's tilt on the tilted paraxial landing.  Pinned comparatively (the
  argument must CHANGE the answer, and by an amount consistent with the kernel
  difference rather than a blunder).
* **`_sphere_parab_conversion` takes `dy`**; `dy=None` / `dy=dx` is
  `np.array_equal` to the pre-fix output, and `dy=3dx` matches a hand-built
  eikonal exactly.
* **`CarrierField` freeze**: see §5 -- shipped as the ANNOUNCEMENT half of a
  deprecation cycle rather than as a hard freeze, because a live consumer
  outside my ownership mutates the attribute.
* **Self-contradicting comments corrected** in `_carrier_step_fast` and
  `propagate_carrier_referenced` (see the changelog for the exact statements).
  The "physically correct transfer function" claim is now stated as what the
  derivation supports, noting the audit's own measurement that the exact and
  Fresnel kernels agree to ~1e-4 relative on a real carrier leg.

### 2.5 Cost of the C1 diagnostics

Honest accounting, since the guard runs on the chain's default landing.  The
readout now measures the input envelope's centroid, amplitude radius and
residual curvature, and the stop envelope's centroid and radius.  Priced against
the readout leg (medians of 9 runs):

```
N      readout    C1 adds      as % of the pre-C1 readout
512     96.3 ms    33.6 ms         39 %   (fit 24.1 + stop centroid/radius 9.5)
1024   364.8 ms    80.8 ms         28 %   (fit 46.0 + 34.8)
2048  1459.9 ms   248.1 ms         21 %   (fit 100.8 + 147.3)
```

Three lossless reductions were applied to get there (they were worth ~2x):
the fit is computed ONCE and threaded into both consumers (it was being done
twice); `_default_focus_standoff` and `_near_focus_needs_bridge` accept the
centroid/radius the caller already holds; and `_fit_carrier_inv` /
`_envelope_amp_radius` broadcast instead of building a `meshgrid` (bit-identical,
verified).  Beyond that, the diagnostic fit reads at most
`_FIT_CARRIER_DIAG_MAX_LINES = 512` lines per axis -- which does NOT change the
sample pitch along the differenced axis (the basis of the increment estimator's
exactness), only how many lines the weighted average runs over.  Grids at or
under 512^2 take stride 1, i.e. bit-identical to the unstrided fit; above that
the measured worst deviation is

```
centred smooth envelope (parabola / r^4 / coma / astigmatism)   6.5e-09
decentred 0.6 waists, same aberrations                          1.6e-05
decentred 1.5 waists, same aberrations                          1.8e-04
+ 30 % PER-PIXEL uncorrelated amplitude noise                   2.6e-03
```

all at stride 8 (the shipped stride is 4 at N = 2048).  The number is used as the
SMALL term of `1/R_eff = 1/R + 1/R_env`, where on the C1 fixture 0.3 % of it is
0.03 % of `1/R_eff`, against a standoff plateau on which the derivation's own
matrix has M = 2.8..3.6 all landing inside 6.1e-3..9.2e-3 of readout error.  The
public `carrier_referenced_fit_radius` never strides.

Against the C4 gains the net is comfortably positive: a carrier leg went 4.00 ->
2.00 grids, and `carrier_referenced_reconstruct` -- which the readout calls -- is
4.5x faster at N = 2048.

### 2.6 Re-check of everything the audit verified correct

Re-ran the audit's own scripts on the fixed tree; all reproduce their recorded
numbers:

* **Sziklas-Siegman transform** (`p1_gauss.py`): relL2 **1.064e-07 / 1.994e-07 /
  2.655e-07**, phase RMS 7.803e-08 / 1.571e-07 / 2.156e-07 rad,
  `P_carrier/P_analytic = 1.000000` at m = 1.49 / 2.95 / 10.73 -- the audit's
  1.06e-07 / 1.99e-07 / 2.66e-07.
* **Converging carrier through focus** (`p1b_focus.py`): peak ratio 0.993439,
  P = 0.999999 at the focus; unchanged.
* **Paraxial focus readout vs the analytic Gaussian-ABCD oracle**
  (`p3_readout.py`): piston-free relL2 **1.048e-03** (NA 0.05, ext 4),
  **3.808e-03** (NA 0.10), **1.767e-02** (NA 0.05, ext 2, the narrow-grid
  branch); peak ratio 0.99992 / 0.99964; EE(2 w0) 0.99964 vs 0.99964 -- the
  audit's 1.05e-03 / 3.81e-03 / 1.77e-02.
* **`reconstruct`/`envelope` round trip and the aliased-carrier design**
  (`p2_alias.py`): round trip 4.403e-09, zero warnings at 4.796 rad/px, the
  continued-leg control identical to the field (7.511e-03 both arms).
* **`R -> inf` guards, `_rereference` no-op, aperture transmission**
  (`p4_fit.py` §4-5): `dx_out/dx = 1.000000000000` at `R = inf / 1e12 / 1e15`,
  `propagate(R=1e15)` vs `R=inf` 4.841e-15, transmission 0.721850 == the retained
  power fraction, `refit_carrier=True` returns R unchanged.
* **A two-group chain against the brute-force ASM + `apply_real_lens_traced`
  arm** (`p5_chain.py`, point source 30 mm -> singlet -> 40 mm -> singlet ->
  half the image distance, N = 2048): power ratio **1.000067**, r2m **1.13882 mm
  vs 1.14389 mm (0.44 %)**, centroids on axis to 1.1e-10 m on the chain arm and
  3.1e-09 m on the brute arm -- the audit's 1.000067 / 0.44 % exactly.
* **Sign conventions, `_check_readout_replica` V3 geometry, `_tilt_obliquity`,
  DOE book-keeping, the astigmatic piston, no module-level mutable caches**:
  covered by the existing suite, which passes (§4).

### 2.7 Backend twins (COMMON §9)

**JAX** (0.10.1, `jax_enable_x64` on, as the campaign convention requires): every
changed NumPy path with a backend twin agrees with it at float64 round-off.
Measured -- `_radial_carrier_phase` numpy-builder vs jnp-builder 3.5e-16 (centred
and decentred); `_rereference` on a JAX field **0.0**; a whole carrier leg
(`propagate_carrier_referenced`, which routes JAX through `_exact_tf_2d_xp` and
`_tf_phase_to_H`) 4.7e-16 / 5.2e-16 / 5.6e-16 relative at `R = inf / 0.05 /
-0.02 m` with `dx_out` identical to 12 digits; `_asm_axis` at N = 64 / 65
**0.0** / 3.0e-16 (the odd-N band-limit fix is on the `bld` axis, so BOTH arms
move together); and the C1 containment guard on a JAX field reads the identical
containment (3.212815) and field (4.9e-16).  Pinned in
`TestC5SmallerItems::test_the_backend_twins_agree_with_the_numpy_paths`.

**CuPy** is not installed on this machine (`ModuleNotFoundError`), so it is
desk-checked, and one change was made because of that desk-check: the two
host-side reductions the C1 guard adds now pull through `backend.to_numpy`
rather than `np.asarray`, which raises on a CuPy device array (implicit transfer
is blocked) -- the same rule `_envelope_amp_radius` and `_fit_carrier_inv`
already follow.  The separable screen builds use only `bld.exp` and
broadcasting, both of which CuPy provides; the `cos`/`sin` transfer-function
build is confined to the `xp is np` arm of `_tf_phase_to_H` and to
`_exact_envelope_tf_step`, which is NumPy-only.  Consequence, stated plainly:
the CuPy / JAX kernel `_exact_tf_2d_xp` gets the `cos`/`sin` win only through
`_tf_phase_to_H`'s device branch, which already built `H` that way for
complex64; its complex128 arm still takes `bld.exp`.  That is no regression, and
a follow-up rather than a defect.

---

## 3. Files touched

**Modified (owned):**

* `lumenairy/propagators/carrier.py` (+822 / -82 lines)
* `lumenairy/propagators/carrier_field.py` (+116 / -16 lines)

**Tests modified (owned -- carrier test files):**

* `tests/unit/test_carrier_field.py` -- mixed-wavelength fixture built rather
  than mutated; new deprecation test.
* `tests/unit/test_niche_c5_exact_tilted_reference.py` -- the
  `_radial_carrier_phase` byte-identity pin re-barred onto the derived bound
  plus the fail-before switch.
* `tests/unit/test_niche_r9_highna_final_leg.py` -- the paraxial fail-before arm
  waives `on_focus_containment` and now asserts the un-waived call is refused.
* `tests/unit/test_niche_c1_consolidation.py` -- `_run_skew(broken=True)` waives
  the guard on the fail-before arms only; the focus-readout whitelist fixture
  gains the new key.
* `tests/unit/test_niche_d3_guards.py` -- the shared `_RO` readout options waive
  `on_focus_containment` on the multi-congruence FAN fixture (see §4.1.6).
* `tests/unit/test_niche_d5_dx_flatness_gate.py` -- the `focus_readout` dict
  waives it for the `paraxial_leg` TEETH arm (see §4.1.7).

**New:**

* `tests/unit/test_audit2609_a6_carrier.py` (995 lines, 80 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A6_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A6_CHANGELOG.md`

No other file was edited.  No git write command was run.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a6_carrier.py` | **80 passed**, 56 s |
| `pytest` on the 9 fast carrier files (`test_carrier_referenced`, `test_fix_v1_v8_readout_guard_and_standoff`, `test_fix_v10_decentred_standoff`, `test_mixed_precision_carrier_helpers`, `test_niche_d14_deterministic_carrier_fit`, `test_niche_r6_auto_carrier_fit`, `test_niche_c9_sphere_parab_exact_conversion`, `test_niche_c14_encapsulation`, `test_niche_audit_w4_input_kind`) | **386 passed, 2 failed**, 263 s -- both failures in `test_niche_audit_w4_input_kind` and NOT mine (see below) |
| `pytest` on the 14 heavy carrier/chain files (`test_carrier_field`, `test_niche_tight_focus_readout`, `test_niche_r8_tiltaware_chain_api`, `test_niche_d1_tilted_carrier`, `test_niche_k2_carrier_backends`, `test_niche_s8_sphere_carrier_reference`, `test_niche_c5_exact_tilted_reference`, `test_niche_exact_gap_kernel`, `test_niche_perf_round2_2026_08_10`, `test_niche_r9_dx_scaling_fix`, `test_niche_r9_highna_final_leg`, `test_fix_tilt_quadratic_opl`, `test_niche_s10_sibling_patterns`, `test_niche_p6_astigmatic_aperture`) | **376 passed, 4 skipped, 3 failed** on the first pass; all three addressed (see below), re-run **green** |
| `pytest` on the 6 chain/multi/pipeline files (`test_niche_d2_chain_multi`, `test_niche_p8_capstone`, `test_niche_c1_consolidation`, `test_hammer_h6_traced_carrier_eikonal`, `test_pipeline`, `test_carrier_field`) | **175 passed, 3 failed** on the first pass; all three addressed, re-run **green** |
| **confirming sweep** over all 29 files above plus `test_v4_15_3_dispatcher_pin_2d_scalar_field` | **757 passed, 4 skipped, 0 failed**, 2514 s (41:53) |
| second sweep over the 21 remaining files that import from `propagators.carrier` (`test_niche_d3_guards`, `test_niche_d5_dx_flatness_gate`, `test_niche_d6_exact_tilted_leg`, `test_niche_e4_corrected_relay_oracle`, `test_niche_d4_dgrating`, `test_niche_p2_design_battery`, `test_niche_c3_gap_paraxial_guard`, `test_niche_newton_pool_both_fits`, `test_niche_gap_frame_observable`, `test_niche_d8_congruence_workers`, `test_niche_c15_inverse_map`, `test_fix_newton_pool_memory`, `test_niche_d9_grid_origin`, `test_verify_perf_fixes_2026_08_10`, `test_niche_p2_guards`, `test_niche_audit_w9_dispatch2`, `test_fix_d5_fit_domain_basis`, `test_niche_s12_rs_fine_clamp_warning`, `test_niche_audit_p2_fresnel_tf_buffer`, `test_fix_grid_intent_override_2026_08_10`, `test_pipeline_spec_guard_validity`) plus `test_niche_audit_w4_input_kind` | **818 passed, 6 skipped, 14 failed**, 1642 s (27:21).  3 were mine (the containment guard on two more deliberate fail-before fixtures); addressed, `test_niche_d3_guards` + `test_niche_d5_dx_flatness_gate` re-run **54 passed**, 586 s.  The other 11 are NOT mine -- see §4.2. |
| final re-run of every test file this WP edited, together (`test_audit2609_a6_carrier`, `test_carrier_field`, `test_pipeline`, `test_niche_c5_exact_tilted_reference`, `test_niche_r9_highna_final_leg`, `test_niche_c1_consolidation`) | **246 passed**, 452 s |

There is no `validation/` topic file for the carrier chain: `grep -rln
'carrier_referenced\|propagate_traced_carrier\|CarrierField' validation/*/test_*.py`
returns nothing, so `validation/run_all.py` has no leg that exercises these
modules.  The repository-level coverage of them is entirely in `tests/unit/`,
which is what the two sweeps above run.

The repro scripts (before and after, all under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/CARRIER/`):
`p6c_mismatch.py`, `p6_invariance.py`, `p4_fit.py`, `p7_odd_dtype.py` (minus the
deleted-name import), `p8_c64chain.py`, `p10_perf.py`, `p10b_tf.py`,
`p1_gauss.py`, `p1b_focus.py`, `p2_alias.py`, `p3_readout.py`, `p5_chain.py`.

### 4.1 Failures found and what they were

1. `test_niche_c5_exact_tilted_reference.py::test_the_sphere_parabola_conversion_is_untouched`
   -- asserted `np.array_equal(_radial_carrier_phase(...), np.exp(...))`, i.e.
   the whole-grid build bit for bit.  This is a test pinning the pre-C4
   arithmetic.  **Fixed in place** (COMMON §verification-bar): it now asserts the
   derived 1e-11 rad bound AND the byte-identity behind
   `_SEPARABLE_CARRIER_PHASE = False`, so the pin survives as the fail-before
   switch.  Note its FIRST assertion -- that `_sphere_parab_conversion` is
   untouched -- passes, which independently confirms the `dy` change is
   bit-identical on a square grid.
2. `test_niche_r9_highna_final_leg.py::test_r9_exact_leg_focuses_highna_sphere`
   (2 params) -- the new containment guard refused the test's PARAXIAL arm.  That
   arm is the test's deliberate fail-before (`assert ee_par < 0.10`), and the
   guard measured containment 1.055 / 0.919 on an input grid holding 2.56 / 2.19
   beam radii.  The guard and the assertion are measuring the same failure from
   two directions, so the arm now waives the guard explicitly (as it already
   waives `on_replica`), and the test additionally asserts the un-waived call IS
   refused.
3. `test_niche_c1_consolidation.py::test_breaking_the_tilted_path_...[no tilt ramp]`
   -- same class: a fail-before arm that monkeypatches `_tilt_ramp` to `None`,
   whose envelope then reaches the stop plane at containment 0.480.  Waived on
   the broken arms only (`_run_skew(broken=True)`), not in the shared fixture.
4. `test_niche_c1_consolidation.py::test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes`
   -- the whitelist is `{dx_out, N_out, centre_out} | _OUTPUT_GRID_PASSTHROUGH`
   and the test enumerates every key; `on_focus_containment` added to both.
5. `test_pipeline.py::test_batched_aggregation_matches_one_call_to_float64` --
   `validation/pipeline/driver.py:462` does `acc.envelope += ...`, which a hard
   `frozen=True` breaks at the re-bind.  Resolved by shipping the freeze as a
   deprecation cycle instead (§5); the test passes unchanged.
6. `test_niche_d3_guards.py::test_multi_congruence_input_warns_and_names_the_route`
   and `::test_multi_congruence_ignore_reproduces_the_pre_d3_silence` -- the
   containment guard refused the FAN fixture (two well-separated congruences on
   one grid, whose second moment is taken across the SEPARATION: containment
   0.867 on an input the guard reads as 0.72 beam radii wide).  The refusal is
   not wrong -- a multiplexed fan is precisely the "populated, credible-looking,
   scrambled" answer that file's own D3 gate exists to catch -- but it would
   pre-empt the subject of those tests, which is the D3 WARNING and its wording.
   `_RO` now waives it, with the reasoning written at the fixture.
7. `test_niche_d5_dx_flatness_gate.py::test_gate_has_teeth[paraxial_leg-...]` --
   same class again: the `paraxial_leg` TEETH arm is a deliberately reverted
   configuration (measured FWHM 8.58 um / EE2 8.4 % against 2.755 um / 63.9 %)
   whose envelope reaches the stop plane at containment 0.963.  Waived in the
   `focus_readout` dict, next to the `on_replica` waiver already there for the
   same arm.

### 4.2 Failures NOT caused by this WP

**11 of the 14 in the second sweep**, all in files under concurrent edit by
other work packages (`git status` shows `lumenairy/elements/_lens_traced.py`,
`_lens_real.py`, `_lens_imap.py`, `_lens_jax.py` and six more modified):

* `test_fix_newton_pool_memory.py` (5) and `test_niche_newton_pool_both_fits.py`
  (3) -- `could not locate _invert_newton_parallel`, `worker payload is missing
  'newton_fit'`, and the `_script_has_main_guard` AST-detector cases.  All three
  names live in `lumenairy/elements/_lens_traced.py` (WP-A3), which is mid-edit.
* `test_verify_perf_fixes_2026_08_10.py::test_capstone_stage_b_is_import_safe_and_blanket_free`
  -- `capstone_stageB.py has no top-level __main__ guard`, via the same
  `_lens_traced._script_has_main_guard`.
* `test_niche_d6_exact_tilted_leg.py::test_decentred_carrier_decentre_penalty_envelope`
  (`the ON-AXIS path regressed: EE2 ratio 0.9698` against a 0.97 bar and a
  2026-07-29 measurement of 0.9966) and
  `test_niche_p2_design_battery.py::test_battery_through_focus_unclipped_doublet_matches_gaussian`.
  **Exonerated by measurement, not by assertion**: re-run with WP-A6's only
  non-bit-identical change on that path reverted in process
  (`_SEPARABLE_CARRIER_PHASE = False`, via a session plugin), both produce the
  IDENTICAL numbers -- EE2 ratio 0.9698 and `(2.05e-05, 1.741296086465325e-05)`
  to every printed digit.  Every other WP-A6 change reachable from
  `final_leg='exact'` is verified bit-identical (`_exact_envelope_tf_step`,
  `_envelope_amp_radius`, `_fit_carrier_inv` at the default centre and stride 1,
  `_sphere_parab_conversion` on a square grid) or elementwise-identical
  (`_fourier_upsample_crop`'s in-place rescale).  Both tests run
  `apply_real_lens_traced`, i.e. the file WP-A3 is rewriting.

**And 2 in the first fast batch:**
`tests/unit/test_niche_audit_w4_input_kind.py::test_wired_site_declares_expected_input_kind[beam_stats.py::beam_d4sigma(E)->field]`
and `::test_all_sixty_eight_sites_are_wired` -- both say
`lumenairy/analysis/beam_stats.py` now has 2 `_check_2d_scalar_field` calls where
the pin expects 1 (69 -> 70 repo-wide).  `git diff --stat HEAD` shows
`beam_stats.py` at +50/-9 from another WP; my diff adds **zero**
`_check_2d_scalar_field` calls (`git diff HEAD -- lumenairy/propagators/carrier.py
| grep -c '^+.*_check_2d_scalar_field'` = 0).  Owner: whoever owns
`analysis/beam_stats.py` (WP-A7).  (They had been fixed by the time of the
second sweep, where that file passes.)

**Two non-reproducing failures, both attributable to concurrent edits.**

* An early run of the fast batch showed 7 failures in
  `test_niche_d14_deterministic_carrier_fit.py` that did not reproduce on any
  later run (and the file passes alone, 18/18).  Two other agents' files were
  mid-edit at that moment -- I hit a live `SyntaxError` in
  `lumenairy/_cache_registry.py` and an `AttributeError` from
  `elements/pmm/twod.py` in the same window.
* The first pass of the final 6-file re-run showed 1 failure in
  `test_niche_c1_consolidation.py::test_the_measured_na_guard_closes_the_paraxial_pre_checks_blind_spot`;
  the identical command re-run immediately after gave **246 passed**, and the
  test passes alone in 7 s.  That test asserts literal digit strings
  (`'2.110 %'`) taken from `apply_real_lens_traced`'s `na_exit` statistic, and
  its own docstring is dated TODAY ("RESTATED 2026-09-12 (audit T13)") -- i.e.
  it is a moving target while WP-A3 rewrites that statistic.  Nothing in this
  WP feeds it.

Neither is test-order coupling in my changes: WP-A6 adds no process-global
mutable state beyond the module flag `_SEPARABLE_CARRIER_PHASE`, and every site
that flips it (one test in this WP's file, one in
`test_niche_c5_exact_tilted_reference.py`) restores it in a `finally`.

---

## 5. Requested changes outside my ownership

1. **`validation/pipeline/driver.py:462`** -- change
   `acc.envelope += res.field.envelope  # bit-identical to aggregate()` to
   `np.add(acc.envelope, res.field.envelope, out=acc.envelope)`.
   *Why:* `CarrierField` is on a deprecation cycle to `frozen=True` (audit C5;
   `CarrierSpec` and `FieldGrid` are already frozen).  `+=` on an ndarray
   attribute mutates the array and THEN rebinds the attribute for no reason; the
   rebinding is what a frozen dataclass refuses.  `np.add(..., out=...)` is the
   same arithmetic bit for bit, needs no rebind, and is already frozen-safe.
   This is the only consumer in the repository that mutates a built
   `CarrierField` (`grep -rn` over `lumenairy/`, `tests/`, `validation/`,
   `examples/`, `scripts/`).
   *Status:* because no WP owns `validation/pipeline/`, I did NOT take the hard
   freeze the audit asks for.  Instead `CarrierField.__setattr__` now emits a
   `DeprecationWarning` naming the migration and the horizon
   (`_CARRIER_FIELD_FROZEN_IN = '5.48'`, routed through
   `_deprecation.resolve_removal_version` so it cannot advertise a shipped
   version) and lets the assignment through.  That is what COMMON §8 prescribes
   for removed behaviour, it closes the audit's actual hazard (the bypass is no
   longer silent), and it leaves the suite green.  Once the driver line lands,
   flipping the class to `@dataclass(frozen=True)` is a one-word change --
   `__post_init__` already writes through `object.__setattr__`.
2. **`tests/unit/test_niche_audit_w4_input_kind.py`** (owner: whoever owns
   `analysis/beam_stats.py`) -- `_WIRED_SITES` and the 69-call count need
   updating for the second `_check_2d_scalar_field` call added to
   `beam_stats.py::beam_d4sigma`.  Not mine; see §4.2.
3. **`lumenairy/propagators/mft.py`** (WP-A5) -- `angular_spectrum_propagate_mft`
   returns complex128 for a complex64 input.  That is why the chain's TILTED
   paraxial landing still returns complex128 after C3 (the chief-ray ramp it
   multiplies in is a complex128 array on the readout grid).  Small, because the
   array is `N_out^2` rather than the chain grid; worth doing when that module is
   next opened.

---

## 6. Deferred

### 6.1 §15.9 / CARRIER "Alternative algorithms" 1-2 -- Collins / ABCD-Fresnel transport with a Bluestein output grid

**Not attempted.**  The WP gates it on being able to score it against the
existing chain on the CARRIER fixtures with a derived tolerance; the honest
position is that the fixtures needed for that gate (the design-121 assets in
`validation/repro_traced_carrier_122/`) are untracked on this machine, and the
independent fixtures I have exercise only the low-NA on-axis surface, which is
the surface on which the two transports agree by construction.  Landing it
un-gated would replace a measured transport with an unmeasured one.

**Design, concretely.**  Collins (1970, *JOSA* **60**, 1168): for an ABCD system
`E_out(x) = (i/(lambda B)) integral E_in(u) exp(-i k (A u^2 - 2 u x + D x^2)/(2B)) du`.
Factor it as chirp x chirp-Z x chirp:

1. `g(u) = E_in(u) exp(-i k A u^2/(2B))` -- one separable screen, which
   `_radial_carrier_phase`'s new outer-product build already provides.
2. a chirp-Z (Bluestein) of `g` onto the chosen output lattice -- exactly the
   transform `propagators/mft.py::angular_spectrum_propagate_mft` already
   performs, with the separable variant this module already ships
   (`_EXACT_READOUT_SEPARABLE_BLUESTEIN`, measured 2.4-6.7x there).
3. `E_out(x) = (i/(lambda B)) exp(-i k D x^2/(2B)) * (2)` -- a second separable
   screen.

For a quadratic carrier this is *exactly* the Sziklas-Siegman result with the
output pitch chosen FREELY instead of forced to `m dx`.  The consequence is that
`m -> 0` stops being a singularity, which removes the entire near-focus
apparatus: `_near_focus_needs_bridge`, `_propagate_carrier_focus_crossing`,
`_axis_bridge`, `_default_focus_standoff`, `_small_extent_focus_standoff_f`, the
replica guard's standoff coupling -- and C1 itself, since there is no contracted
co-moving grid to clip the beam against.  Cost: 3 FFTs of
`next_fast_len(N + N_out - 1)` instead of 2 of `N`, i.e. ~2-3x per leg before the
separable Bluestein's own 2.4-6.7x is applied.

**Effort and the gate it must pass.**  ~4-6 days.  Ship behind
`transport='sziklas' | 'collins'` (default `'sziklas'`).  The acceptance gate
should be, in order: (a) an analytic Gaussian-ABCD oracle at NA 0.03-0.45 and
grid extents 1.5-10 w, requiring `'collins'` to be no worse than `'sziklas'` in
every cell and materially better in the cells the small-extent branch exists for;
(b) the C1 mismatch matrix above -- `'collins'` should read peak ratio 1.0000 at
every R/R0 because the output pitch no longer depends on the carrier at all;
(c) a two-group chain against the brute-force ASM + `apply_real_lens_traced` arm
the audit already built (`p5_chain.py`: power 1.000067, r2m 0.44 %), requiring
agreement at least as close; (d) `_multi` K = 1 vs K = 2 exactness.  The sampling
conditions to write the guard against are in Kelly, *Appl. Opt.* **53**, 2861
(2014), not in the geometric-margin model `_FOCUS_STANDOFF_MARGIN` uses.

### 6.2 Not in scope, recorded

* **Decentred `_rereference`** (see C2's residual risk): the aperture refit and
  the re-reference screen are a matched origin-centred pair.  Giving both a
  `centre=` would let `carrier_referenced_aperture(refit_carrier=True)` be
  correct on a decentred beam.  ~1 day including tests.
* **The module split** (CARRIER's P3 structure entry: six modules, no new import
  cycles, seams identified).  Deliberately not attempted here -- it would make
  every other WP's diff on this file unmergeable.  The audit's table is still
  accurate after this WP; the only change is that `carrier_field.py` now also
  imports `_phasor_rows`, which belongs in the proposed `carrier_core.py`.
* **`_multi_worker_run` warning re-categorisation** and the **serial
  `readout_tile='auto'` probe pass** (CARRIER "smaller items"): both are in
  `propagate_traced_carrier_chain_multi`, both are behavioural changes to the
  parallel orchestrator, and neither is reachable by a test that fits this WP's
  budget (a K-congruence run is ~30-90 s per leg here).  Recorded, not done.
* **The `_BRIDGE_ZR_FACTOR` sweep** the audit lists under "unverified
  suspicions" (piston fidelity through the focus-crossing bridge, +1.65e-02 rad
  on-axis).  Not a finding; not touched.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A6_CHANGELOG.md`

# VERIFY-WP-A6 -- independent re-verification of the traced-carrier chain fixes

Subject: commit `a18ab074` ("fix(carrier): WP-A6 ..."), base `a18ab074^`, on
`lumenairy/propagators/carrier.py` and `carrier_field.py`, against audit rows
C1-C5 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.4 and the partition
report `.../CARRIER.md`.

Everything below is a MEASUREMENT made by the verifier on this machine with
`OPENBLAS_NUM_THREADS=1`, on Python 3.14.6 / numpy 2.4.6 / jax 0.10.1.  Where
the engineer's report quotes a number I re-ran its script and quote mine next
to it.  Where a finding is a P1 I built a second fixture the WP did not use
(λ = 0.85 µm and 0.633 µm rather than 1.31 µm; NA 0.08; odd and anamorphic
grids) and an oracle the library did not produce (an analytic Gaussian-ABCD
`q`-parameter field, a dense numerical scan of the containment inequality,
`np.fft.fftfreq`, a hand-written transcription of the PRE-FIX moment
estimator).

---

## 1. Verdicts

| # | Verdict | What I re-measured | Result |
|---|---|---|---|
| **C1** | **VERIFIED-WITH-NOTES** | `repro/CARRIER/p6c_mismatch.py`, `p6_invariance.py`; a brute-force scan of the containment inequality (4e5 samples + 200 bisections); an analytic Gaussian-ABCD focal oracle at λ 0.85 µm / NA 0.08; a 910-run lengthen-only sweep incl. top-hat / two-lobe / decentred / noisy envelopes and residual curvature of both signs | headline reproduces to every printed digit; closed form == brute-force boundary to **1.25e-14** relative; resolver never shortens (0/910 runs) and never moves a real-amplitude envelope (0/650); **two gaps found** -- one FIXED here (OI-A, the `alpha <= 0` early return), one recorded (OI-1, grids narrower than 3.2 beam radii) |
| **C2** | **VERIFIED** | `repro/CARRIER/p4_fit.py`; a hand-written transcription of the pre-fix estimator; tilt+curvature together; top-hat / clipped / two-lobe / triangular envelopes; `refit_carrier=True` on a decentred aperture | `'origin'` is the pre-fix arithmetic **bit for bit** (12/12 cells); `'auto'` recovers the constructed radius to **2.2e-16** with tilt AND curvature present, where `'origin'` reads 6.8 mm..450 mm for a 50 mm truth; `refit_carrier` is self-consistent, not a latent C2. **One residual found** (OI-2, the sub-pixel snap) |
| **C3** | **VERIFIED** | every public entry point of the chain, both readouts, the `CarrierField` verbs, float32 real input, JAX with x64 on and off; `repro/CARRIER/p8_c64chain.py` | complex64 preserved everywhere I could reach, INCLUDING the tilted paraxial readout and the exact readout; the chain-level residual the report records was found to be real and is now **FIXED** here (OI-B, coordinator item 2). One default change under-documented (OI-4) |
| **C4** | **VERIFIED-WITH-NOTES** | independent whole-grid oracles at N up to 8192; `tracemalloc`; `np.fft.fftfreq`-based transfer-function oracles at odd N and with tilt | allocation claims reproduce (3.50 -> **1.01** grids, TF step **2.00** grids, reconstruct 3.50 -> **2.00**); the regrouping is exact-in-exact-arithmetic and lands at **1.0-1.6 x eps·\|arg\|**. **The bar and its derivation were wrong** (claimed "three orders below the ~1e-11 representation noise"; it is AT the floor, and a fixed 1e-11 bar fails above ~3e4 rad of argument) -- corrected here in the code comment and in three tests |
| **C5** | **VERIFIED-WITH-NOTES** | `np.fft.fftfreq` oracles at N = 64/65/127/128/129/1025 with the band limit on AND off; a stable hand-built eikonal; a monkeypatch SPY on the inner carrier leg; the full deprecation cycle incl. pickle/deepcopy/replace; `validation/pipeline/driver.py` | `_asm_axis` in register at every parity (<= 5.9e-14, identical with the mask on and off); `gap_kernel`/`tilt` verifiably ARRIVE at the inner call; deprecation cycle complete and construction silent; the driver line is bit-identical and warning-free. **Two claims found false**: the `fftfreq` "exactly" (docstring corrected here) and `dy=` reaching no caller (recorded as OI-3); one ruff F821 FIXED here (OI-C, coordinator item 1) |

**Nothing regressed.**  Every number the audit listed under *checked and found
correct* reproduces (§5).

Three defects in WP-A6's own files were found and fixed by this verification
(§3); five open items are recorded for the orchestrator (§6).

---

## 2. Per finding

### C1 -- the focus readout sized its stop grid from the carrier

**Headline reproduces exactly.**  `repro/CARRIER/p6c_mismatch.py`, shipped
defaults, one physical field, only the reference carrier varied:

```
R/R0   standoff        half/beam   peak ratio   warnings      report claims
1.00    222.39 um        3.21       1.000000        0        222.4 / 3.21 / 1.000000
0.99   1008.60 um        3.20       0.999514        0       1008.6 / 3.20 / 0.999514
0.98   1873.95 um        3.20       0.998602        0       1874.0 / 3.20 / 0.998602
0.95   4173.27 um        3.20       0.994304        0       4173.3 / 3.20 / 0.994304
0.90   7144.93 um        3.20       0.985236        0       7144.9 / 3.20 / 0.985236
```

`p6_invariance.py` likewise: relL2 **5.139e-04 / 7.603e-03 / 3.181e-02** at
R = -23 / -19.6 / -18 mm, peak ratios **0.999898 / 0.998602 / 0.985236**,
0 warnings -- the report's 5.1e-04 / 7.6e-03 / 3.2e-02 and the same three peak
ratios.

**The closed form is right, and I checked the algebra numerically rather than
reading it.**  I re-derived the quadratic independently (squaring
`half·|ζ−ζ_cf|/ζ_cf ≥ M·w_env·sqrt((1+cζ)² + (ζ/zR)²)` gives exactly the
shipped α, β, γ) and then compared its root with the boundary found by a dense
scan plus 200 bisections of the *inequality itself*, over 3 radii × 2 widths ×
3 extents × 6 residual curvatures:

* worst relative disagreement over the 28 cells where both are defined:
  **1.253e-14**;
* the claim the root SELECTION rests on -- `q(ζ_cf) < 0`, so the carrier focus
  is never inside the containment set -- holds as an identity, not a sample:
  over 200 000 log-uniform cells the ratio `q(ζ_cf)/|−Q[(1+cζ_cf)²+(ζ_cf/zR)²]|`
  is **exactly −1.000**, and the identity `q(ζ_cf) == −Q[...]` holds to
  **0.051 ×** (64 eps × the dominant term), which is the cancellation floor
  (the `half²` terms sum to zero exactly, so a relative tolerance is not the
  achievable statement here).

**"The resolver can only lengthen" holds, on and off its calibration matrix.**
130 (NA, ext) cells (NA 0.01..0.45 × ext 0.8..16, i.e. well outside the
resolver's own 6 × 10 matrix) × 7 envelope kinds = 910 runs (flat Gaussian,
+10 % noise, top-hat, two-lobe, decentred 0.8 w, residual curvature +2.5 and
−2.5 1/m):

* cells where the new leg is SHORTER than the pre-fix leg: **0**;
* cells where a real-amplitude envelope moved at all: **0** (the beam term
  short-circuits on `inv_env == 0.0`, which a real envelope gives exactly);
* cells that lengthened: **81**, all of them the two curved kinds -- so the
  term is live in both signs of residual curvature, not only the audit's.

**An independent oracle at a different wavelength and NA.**  λ = 0.85 µm,
NA = 0.08, w = 0.6 mm, N = 1024, ext = 4, scored against an analytic Gaussian
ABCD focal field (amplitude, curvature and Gouy phase) written in the test file:

```
R/R0   peak vs the ORACLE (post -> pre)   piston-free relL2 (post -> pre)
1.00      0.999799 -> 0.999799               2.41e-03 -> 2.41e-03
0.99      0.995468 -> 0.891233               1.57e-02 -> 6.08e-02
0.97      0.973251 -> 0.216767               4.27e-02 -> 5.54e-01
0.93      0.919521 -> 0.024932               8.54e-02 -> 8.37e-01
0.90      0.885027 -> 0.002060               1.10e-01 -> 1.00e+00
```

The fix is worth a factor 4.5 to 430 in peak on this fixture.  Note that the
post-fix residual is materially larger here than the 0.9995..0.9852 the report
quotes at NA 0.05: the leg the fix resolves is longer and the hand-off model
error scales as ~NA³.  That is a scope fact worth having in the record, not a
defect.

**Gap A (FIXED here).  `alpha <= 0` returned 0.0 and called the margin
"unreachable at ANY leg length".**  That is only true when the input plane is
also uncontained.  With `alpha < 0` the parabola opens downward and the
containment set is the closed interval BETWEEN the roots; when
`gamma = q(0) >= 0` the input plane is inside it and the larger root is a
perfectly good stop plane.  `alpha = half²/ζ_cf² − Q(c² + 1/zR²)` flips sign
with the carrier mismatch, so this is reachable from the same fixture family
the finding is about.  Measured on R = −20 mm, w_env = 200 µm, ext = 6,
residual 1/R_env = −60 1/m, λ = 0.85 µm:

```
                     leg        containment (meas / model)   guard      peak
before this fix   1705.9 um        0.866 / 0.500            REFUSES    1.408x low
after             5929.9 um        3.1996 / 3.2000          silent     reference
brute-force boundary  5929.9 um  (the closed form now agrees to 1.2e-11 relative)
```

Fixed in `carrier.py:_beam_containment_standoff` by removing the `alpha > 0`
gate: `(-beta - sqrt(disc))/(2*alpha)` is the smaller root when the parabola
opens up and the LARGER when it opens down, so one line serves both, with
`gamma < 0 and alpha <= 0` (nothing to resolve) kept as the only early return.
Provably a no-op wherever `alpha > 0`, and wherever the envelope is real.
Pinned with its fail-before in
`test_audit2609_a6_verify_carrier.py::TestVerifyC1Quadratic::test_the_downward_quadratic_still_resolves_a_leg`.

**Gap B (recorded, OI-1).  Below 3.2 beam radii of grid the resolver is
inoperative.**  `gamma = half² − (M·w_env)² < 0` whenever ext < `_FOCUS_STANDOFF_MARGIN`
= 3.2, i.e. throughout the module's own documented small-extent branch
(V1/V2, ext < M/sat = 3.695).  There is then no margin for the beam term to
resolve against and it returns 0.0.  Measured on the same λ = 0.85 µm fixture
at ext = 3.0:

```
R/R0   standoff (post == pre)   containment   peak vs the oracle   guard
0.99          167.37 um            1.627           0.912560        silent
0.97          312.71 um            0.984           0.300483        warns
0.93         7500.00 um (clamped)  3.000           0.707657        silent
```

so a 1 % carrier mismatch still costs **8.7 % of peak with no warning at all**
(the 1.0 containment floor does not reach 1.627), and the 0.93 row is repaired
only by an accident of the clamp -- `alpha > 0` there, `zeta_minus < 0`, the
resolver asks for more than `|z|` and the readout's `standoff > abs(z)` clamp
turns that into the whole leg.  Whether that accident fires depends on the sign
of `alpha`, which is why 0.97 (worse containment) is repaired less than 0.93.
This is the C1 failure class at reduced magnitude and it is not stated in
WP-A6's report.  Pinned two-sidedly (live at ext 4, inoperative at ext 3) in
`::TestVerifyC1AgainstTheAnalyticFocus::test_a_narrow_grid_is_the_documented_limit_of_the_resolver`.
Concrete design in §6.

**The guard.**  Silent on all 60 cells of the calibration matrix (worst
containment 1.257, worst `window_energy_frac` 0.999863 -- both reproduce), and
it refuses exactly the pre-fix rows the report says it does.  The `1.0` floor
and its ~1.2× two-sided clearance are stated honestly in the code and in the
WP's test; I have nothing to add beyond OI-1.

**Cost.**  The diagnostic stride does not change any default-path answer at or
under 512²: `_fit_carrier_diag_stride((512,512)) == 1`, so the fit is
bit-identical there, and the public `carrier_referenced_fit_radius` never
strides.  Confirmed by the WP's own parametrised test, which I re-ran.

### C2 -- `carrier_referenced_fit_radius` fitted about the grid origin

**`p4_fit.py` reproduces.**  Section 3 (decentred parabola, truth 1.0000):
`increment` reads **1.0000** at 0.5 / 1.0 / 2.0 waists and `gradient`
**1.0002** (the documented `0.5 (dx/w)² = 2.0e-4` central-difference bias, the
same value on axis).  Sections 1, 4 and 5 are unchanged to every printed digit:
`dx_out/dx = 1.000000000000` at R = inf / 1e12 / 1e15, `propagate(R=1e15)` vs
`R=inf` **4.841e-15**, transmission **0.721850**, `refit_carrier=True` returns R
unchanged.

One discrepancy, immaterial: the report's summary table quotes the decentred
pure-tilt after-numbers as `4.8e15 / 2.5e14 / 6.8e13 m`; I measure
`6.64555e15 / -2.72553e14 / 6.35758e13 m` (increment) and
`6.62569e15 / -7.01128e14 / 1.17756e08 m` (gradient, the script's default).  All
are round-off residuals against a truth of `inf` -- the finding verifies either
way -- but the digits do not reproduce, so the table appears to quote an
earlier build.  The report's §2 text ("the smallest post-fix |R| across the six
cases is 1.18e8 m") DOES reproduce exactly.

**Byte-identity checked against something other than the library.**  The WP
asserts `centre='auto' == centre='origin'` on axis, which is circular (both go
through the new code).  I transcribed the pre-C2 estimator from the audit's
quotation of the shipped source and compared: `_fit_carrier_inv` at its default
centre is **equal to the last bit** on 12/12 cells (both estimators × 3
decentres × 2 radii).

**A fixture the WP did not build: tilt AND curvature on a decentred beam.**
Truth R = 50 mm by construction, independent of both:

```
x0/w    L       origin fit    auto fit    explicit fit
0.5   0.002      37.5000 mm   50.0000 mm   50.0000 mm
0.5   0.020       6.8182 mm   50.0000 mm   50.0000 mm
1.0   0.020       7.1429 mm   50.0000 mm   50.0000 mm
2.0   0.020      10.9756 mm   50.0000 mm   50.0000 mm
2.0   0.002      90.0000 mm   50.0000 mm   50.0000 mm
```

`auto` is exact to **2.2e-16** relative at every cell.  Note the cell
`x0 = 1.0 w, L = 0.002`, where the decentre bias and the tilt bias CANCEL and
`origin` reads 50.0000 mm -- a "must differ" assertion would call that a
failure, so my test scores the `origin` arm against the analytic pre-fix
reading `1/R_fit = [(1/R)(w²/2) + L x0]/(x0² + w²/2)` instead (agrees to
3.4e-4).

**Non-Gaussian envelopes** (all decentred by one waist, truth 50 mm):
`auto/R` = 1.0000 (top-hat), 1.0000 (two-lobe), 1.0000 (triangular), 1.0003
(clipped-decentred), 1.0189 (30 % per-pixel noise); `origin/R` = 2.25..3.09 on
the same fields.

**`carrier_referenced_aperture(refit_carrier=True)` is NOT a latent C2.**  The
documented origin-fit pairing is self-consistent, and I checked it on a curved
envelope as well as a flat one, at 0 / 0.5 / 1.0 / 2.0 waists of decentre:

* the **physical field is preserved to <= 2.2e-16 relL2** at every cell --
  `_rereference` applies exactly the screen the radius change asks for, so the
  refit can never produce a wrong field, only a different carrier/envelope
  SPLIT.  That is the property that decides whether this is a defect;
* a flat envelope refits to the input radius to 12 digits and leaves exactly
  zero residual;
* a curved one (residual 2 1/m about its own decentred centre) refits to
  45.455 / 46.876 / 48.388 / 49.451 mm and leaves **zero residual about the
  ORIGIN** (to 1.6e-15) and up to **1.78 1/m about the beam's own centre** at
  2 waists -- i.e. the split is self-consistent but is not the beam's own.

That is exactly what the new docstring says, and the advice it gives (fit with
`centre='auto'`, pass the result as `new_carrier=`) is the right one.

**Residual found: OI-2, the sub-pixel snap.**  The tilt projection is gated on
`centre != (0.0, 0.0)`, and `_envelope_amp_centroid` snaps any decentre under
half a pixel to exactly `(0,0)`.  So a beam decentred by *less than half a
pixel* while carrying a tilt gets neither centring nor projection:

```
x0/dx    L      auto R_fit (truth inf)   centroid snapped?
 0.00   0.02        -4.94e+14 m                yes
 0.20   0.02         0.62502 m                 yes
 0.49   0.02         0.25515 m                 yes
 0.51   0.02         2.48e+14 m                no
 1.00   0.02         8.71e+13 m                no
```

i.e. a 0.26 m radius where the truth is infinite, with a discontinuity at half
a pixel.  Milder than the pre-fix 0.0075 m but the same mechanism.  I did not
change it -- the fix is a deliberate default change (project the tilt
unconditionally) that would move the deterministic-fit pin; design in §6.

### C3 -- complex64 promotion

`repro/CARRIER/p8_c64chain.py` reproduces the claim exactly:
`carrier = TiltedCarrier(inf, L=0.02, ...)` now returns **complex64** where the
audit measured complex128 (the untilted control was and stays complex64).

Re-measured across every entry point I could reach:

```
propagate_carrier_referenced (R=inf / -0.05 / tilted)   c64 -> complex64
carrier_referenced_reconstruct / _envelope / _aperture  c64 -> complex64
carrier_referenced_focus_readout (with and without tilt) c64 -> complex64
carrier_referenced_exact_focus_readout                  c64 -> complex64
CarrierField.full_field / from_full_field               c64 -> complex64
carrier_field.re_reference / aggregate                  c64 -> complex64
CarrierSpec.phasor_on (no dtype=)                       -> complex128 (default kept)
float32 REAL input                                      -> complex128 (unchanged)
JAX x64 ON  complex64 / complex128                      -> complex64 / complex128
JAX x64 OFF complex64                                   -> complex64
```

**The report's attribution of the one remaining promotion was wrong, and the
promotion is now fixed (OI-B).**  §2.3 and §5.3 of WP-A6_REPORT.md blame
`propagators/mft.py::angular_spectrum_propagate_mft`.  WP-A5 measured that
function to preserve complex64, and I measure the readout that calls it to
return complex64 -- so the only complex128 left on that path was the chain's own
chief-ray ramp at `carrier.py:9512`, an ARRAY that a `complex(...)` cast cannot
weaken.  Built now through `_phasor_rows` at the field's dtype: the banded build
is the narrowed whole-grid one **byte for byte** at both parities of the output
grid, and the complex128 branch is the historical expression untouched.  End to
end, through a tilted congruence landing on `final_leg='paraxial'` with a
`focus_readout` window (stage tilt L = 0.018554, i.e. the branch is genuinely
taken): **complex64 in -> complex64 out, complex128 in -> complex128 out**.
The audit's own `p8_c64chain.py` supplies no `focus_readout`, so it lands on the
bare final leg and never exercised that site -- which is why the WP's own repro
showed complex64 while the residual was real.

The C3 request against `propagators/mft.py` in WP-A6_REPORT.md §5.3 should be
**withdrawn**: it is not needed and it names the wrong module.

### C4 -- separable phases, the transfer function, the fine-grid rescale

**The allocation claims reproduce** (`tracemalloc`, medians of 5, this machine):

```
_radial_carrier_phase  N=1024   separable  7.69 ms / 1.02 grids   whole  65.21 ms / 3.50 grids
                       N=2048   separable 33.27 ms / 1.01 grids   whole 316.41 ms / 3.50 grids
_exact_envelope_tf_step N=512                18.11 ms / 2.00 grids
                        N=1024               79.01 ms / 2.00 grids
carrier_referenced_reconstruct N=1024  sep 16.25 ms / 2.00 grids  whole 65.97 ms / 3.50 grids
                                       relL2 sep vs whole = 1.635e-16
_fourier_upsample_crop  n_fine=512                     2.13 fine grids (in place)
```

against the report's 3.50 -> 1.01 / 4.00 -> 2.00 / 3.50 -> 2.00.  Timings differ
(this is a shared box and nothing asserts them, per S1).

**The transfer function is bit-identical** at every shape I tried, and my own
`np.fft.fftfreq`-based hand-built `H` agrees with the library's step to
4.4e-16..6.8e-16 relL2 at (63,63), (64,64), (65,65) and (128,96), untilted and
tilted -- i.e. the untilted short-circuit and the `cos`/`sin` write are exact,
as claimed.

**The bar and its derivation were wrong (OI, corrected here).**  The report,
the module comment and three tests state that the separable regrouping is
"three orders BELOW the float64 representation floor of the arguments these
screens carry (`k r²/2R` reaches 1e5-1e6 rad, i.e. ~1e-11 rad)".  Re-measured
against a whole-grid oracle written in the test:

```
   N     dx      R       max|arg|       eps*|arg|      measured    ratio   <1e-11?
  256   2 um   50 mm   6.2866e+00 rad   1.396e-15     1.422e-15    1.02     yes
 2048   2 um   50 mm   4.0234e+02       8.934e-14     1.138e-13    1.27     yes
 4096   2 um   50 mm   1.6094e+03       3.574e-13     5.684e-13    1.59     yes
 2048   8 um   50 mm   6.4375e+03       1.429e-12     1.819e-12    1.27     yes
 4096   8 um   20 mm   6.4375e+04       1.429e-11     2.183e-11    1.53     NO
 4096  16 um   10 mm   5.1500e+05       1.144e-10     1.746e-10    1.53     NO
```

Three corrections follow, and all three are made:

1. the difference is **at** the argument's own representation floor
   (1.0-1.6 ×), not three decades under it;
2. the representation noise of a 1e6 rad argument is 1.1e-10 rad, not 1e-11;
3. **a fixed 1e-11 bar does not hold over the range the docstring itself
   claims** -- it is exceeded 17× at 5.2e5 rad.

The arithmetic is fine (it is the float64 representation of a phase that
large), so the fix is to the derivation and to the bar: `_radial_carrier_phase`
and `_SEPARABLE_CARRIER_PHASE` now carry the measured table and the scale-free
statement, and the three affected assertions were restated as
`<= 4 eps·max|arg|` (2.5× of headroom at every cell; a regrouping blunder is
O(|arg|), 15 decades up).

`_SEPARABLE_CARRIER_PHASE = False` restores the whole-grid build **byte for
byte** against an oracle written in my test file, at both parities and
decentred.

No test in either file asserts a wall-clock time or a speed-up
(`grep -n 'perf_counter\|time\.\|speedup' tests/unit/test_audit2609_a6_carrier.py`
is empty) -- TESTING_STANDARDS S1 satisfied.

### C5 -- the P3 cluster

* **`_freq_sq_1d` deleted** -- `hasattr(C, '_freq_sq_1d')` is False.
* **`_asm_axis` band limit in register at odd N.**  Against an oracle built on
  `np.fft.fftfreq` (the WP's own oracle restates the expression under test):
  relL2 **<= 5.9e-14** at every parity (exactly **0.000e+00** at several, the
  residual being wavelength-dependent), and *identical with the band limit ON
  and OFF* -- which localises the residual
  to the 2-4 ulp between `fftfreq`'s multiply-by-reciprocal and the module's
  divide, not to the mask.  Pre-fix that mask was half a bin out at odd N.
* **`gap_kernel` and `tilt` verifiably arrive.**  A monkeypatch spy on
  `propagate_carrier_referenced` records exactly one inner call carrying
  `('fresnel', (0.03, -0.02))`.  The WP pins this comparatively; the spy is the
  direct statement.
* **`_sphere_parab_conversion(dy=)`** is correct: `dy=None`/`dy=dx` is
  `np.array_equal` to the pre-fix output and `dy=3dx` matches a numerically
  stable hand-built eikonal to **< 1e-12** (the textbook
  `sqrt(R²+r²) − |R|` form's own cancellation error is 4e-11 rad here, so the
  library is more accurate than the naive oracle).  **But no caller forwards a
  pitch** (all 7 call sites pass one scalar), so a `dy != dx` chain still
  converts its y axis against `dx` -- the latent defect the audit named is
  parameterised, not closed.  Recorded as OI-3.
* **The `CarrierField` deprecation cycle is complete and correct**:
  construction silent; one `DeprecationWarning` per assignment (3 assignments
  -> 3 warnings); `dataclasses.replace`, `with_provenance`, `copy.deepcopy` and
  `pickle` round trips all silent; `_built` is not a dataclass field and not in
  `repr`; `resolve_removal_version('5.48')` -> `'5.48'` against a running
  library at 5.45.1, i.e. the horizon is ahead and resolved through the shared
  resolver rather than advertised as a literal.
* **`validation/pipeline/driver.py:462`** (committed by the orchestrator as
  `04bdbeb1`) reads
  `np.add(acc.envelope, res.field.envelope, out=acc.envelope)`.  `+=` and
  `np.add(..., out=...)` are **byte-identical** on complex128, and the
  `np.add` form raises no `DeprecationWarning` where `+=` raises one -- both
  measured.
* **The comment corrections** are right: `'auto'` really does resolve to
  `'exact'` on NumPy (`np.array_equal(auto, exact)` True, `!= fresnel`), which
  the WP pins.

**One surviving false claim, corrected here (�3).**  `_freq_sq_1d_bld`'s docstring
-- which WP-A6 rewrote -- still said its `ifftshift` is "**exactly**
`(2*pi*np.fft.fftfreq(N, d))**2` for BOTH parities".  Measured: exact at
N = 1/4/5/64/1024 (where `1/(N d)` is representable) and **3 ulp** at
N = 7/65/1025, **4 ulp** at N = 127, **3 ulp** at N = 129; the linear twin
`_freq_1d_bld` 1-2 ulp.  The WP's own test uses a 4-ulp bar, so the test and the
docstring contradicted each other.  Docstring corrected; my test additionally
pins the *property* structurally (the DC bin is at index `N//2` and is the only
exact zero, and the ladder is uniform to `4 eps·max|f|`), which a `- N/2`
offset -- the defect class -- cannot satisfy at odd N.

---

## 3. Defects found in WP-A6's files and fixed here

| ID | File : site | What was wrong | Verification |
|---|---|---|---|
| **OI-A** | `carrier.py:_beam_containment_standoff` | `if not (alpha > 0.0): return 0.0`, documented as "the grid is too narrow for this beam at ANY leg length".  False whenever the parabola opens downward AND the input plane is contained (`gamma >= 0`): the containment set is then the interval between the roots and the larger root is a valid stop plane.  The resolver declined to lengthen where lengthening restores the margin; the guard then refused a readout that is repairable. | brute-force boundary 5929.9 µm vs the new closed form 5929.9 µm (1.2e-11 rel); containment 0.866/0.500 + refusal -> 3.1996/3.2000 + silence; peak 1.408× low -> reference.  Fail-before restores the old gate in process.  `test_audit2609_a6_verify_carrier.py::TestVerifyC1Quadratic::test_the_downward_quadratic_still_resolves_a_leg` |
| **OI-B** | `carrier.py:9512` (chain, tilted paraxial landing) | the chief-ray ramp `exp(i k0 (L u + M v))` was built whole-grid at complex128 and promoted a complex64 readout field at the very end of the chain -- the last C3 site, mis-attributed in the report to `propagators/mft.py`.  An ARRAY, so `complex(...)` cannot weaken it; it has to be built narrow. | banded build == narrowed whole-grid build byte for byte at nn = 32 and 33; end-to-end complex64 -> complex64 and complex128 -> complex128 through a tilted `final_leg='paraxial'` landing with stage L = 0.018554.  `::TestVerifyC3::test_the_tilted_landing_ramp_is_built_in_the_field_s_dtype` and `::test_the_chain_tilted_paraxial_landing_keeps_its_dtype` |
| **OI-C** | `carrier_field.py:469` | `phasor_on`'s complex64 branch passed `_phasor_rows` a lambda closing over `S` and then executed `del S`.  Not a live `NameError` (the helper is eager, and I confirmed the path runs and is correct), but `ruff` reports **F821 undefined name `S`** and the first non-eager builder turns it into one.  Rebound as a default argument. | `ruff check` on both files: **All checks passed** (was 1 error).  Fail-before: the builder `phasor_on` hands over is captured and invoked AFTER `phasor_on` returns -- `NameError` with the closure, correct rows with the default argument.  `::TestVerifyC5::test_the_c64_sphere_builder_survives_its_own_del` |

Also corrected in files I own, all documentation/bar rather than arithmetic:

* `carrier.py` -- the `_radial_carrier_phase` / `_SEPARABLE_CARRIER_PHASE`
  error-floor derivation (measured table, scale-free statement);
* `carrier.py` -- `_freq_sq_1d_bld`'s "exactly ... for BOTH parities";
* `carrier.py` -- the `_beam_containment_standoff` docstring's root-selection
  paragraph (both curvatures, and the one case with nothing to resolve);
* `tests/unit/test_audit2609_a6_carrier.py` -- three `1e-11` bars restated as
  `4 eps·max|arg|` with the re-derivation in the docstring;
* `tests/unit/test_niche_c5_exact_tilted_reference.py` -- the same bar and the
  same re-derivation on the WP's re-barred byte-identity pin.

`ruff check` (project configuration) on `carrier.py` + `carrier_field.py`:
**All checks passed**, F class clean.  `--select E,W` (not in the project's
select list) reports 7 pre-existing findings -- `E741` on the `I = np.abs(E)**2`
idiom and `E501` on long prose lines -- none of them in WP-A6's or my hunks.

---

## 4. Tests

New file: **`tests/unit/test_audit2609_a6_verify_carrier.py`** (72 tests) --
oracles written in the file, fixtures the WP did not use, the three fail-before
arms above, and the two-sided statement of the narrow-grid scope limit.  Its
class map:

| class | what it pins |
|---|---|
| `TestVerifyC1Quadratic` (15) | closed form vs a brute-force scan of the same inequality; the `q(zeta_cf) < 0` identity as an identity; **the `alpha <= 0` fix with its fail-before**; the lengthen-only invariant over 24 (NA, ext) cells × 7 envelope kinds |
| `TestVerifyC1AgainstTheAnalyticFocus` (4) | the readout vs an analytic Gaussian-ABCD field at λ 0.85 µm / NA 0.08, each row with its own measured pre-fix arm; **the narrow-grid scope limit, stated two-sidedly** |
| `TestVerifyC2` (20) | `'origin'` vs a hand-written pre-fix estimator (bit for bit); tilt AND curvature together, with the `origin` arm scored against the analytic pre-fix reading; four non-Gaussian amplitudes; `refit_carrier` field-invariance flat and curved |
| `TestVerifyC4SeparableBound` (6) | the argument-relative bound; **the demonstration that the fixed 1e-11 bar fails inside the documented range**; the flag-off byte-identity against an oracle written here |
| `TestVerifyC5` (21) | `_asm_axis` vs `fftfreq` at 5 parities × 2 axes × band limit on/off; the frequency builders' ulp and the `N//2` offset structurally; **the kernel/tilt SPY**; `dy=` correctness and its unreached callers; **the `phasor_on` closure fix with its fail-before**; the full deprecation cycle; the driver idiom |
| `TestVerifyC3` (6) | every readout on complex64; the `CarrierField` verbs; **the tilted landing ramp, unit and end to end**; the `aggregate` accumulator's float32 cost |

All runs with `OPENBLAS_NUM_THREADS=1 -q --no-header -p no:cacheprovider`, on
the tree carrying my three fixes.

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a6_verify_carrier.py` | **72 passed**, 55 s |
| `pytest` on the 12 files closest to the change (`test_audit2609_a6_carrier`, `test_carrier_field`, `test_niche_c5_exact_tilted_reference`, `test_niche_d5_dx_flatness_gate`, `test_pipeline_spec_guard_validity`, `test_fix_v1_v8_readout_guard_and_standoff`, `test_fix_v10_decentred_standoff`, `test_niche_d14_deterministic_carrier_fit`, `test_niche_r6_auto_carrier_fit`, `test_niche_c9_sphere_parab_exact_conversion`, `test_mixed_precision_carrier_helpers`, `test_carrier_referenced`) | **292 passed, 0 failed**, 722 s (12:02) |
| `pytest tests/unit/test_niche_r9_highna_final_leg.py tests/unit/test_niche_d3_guards.py` (the two guard files WP-A6 re-waived) | **50 passed**, 313 s |
| `pytest tests/unit/test_niche_k2_carrier_backends.py` | **13 passed, 3 skipped** (CuPy absent) |
| `pytest` on the tilt/chain set (`test_audit2609_a6_verify_carrier`, `test_niche_tight_focus_readout`, `test_niche_d1_tilted_carrier`, `test_niche_r8_tiltaware_chain_api`, `test_niche_c1_consolidation`) | **170 passed, 1 failed**, 407 s -- the one failure is COLLATERAL (below) |
| `ruff check` on `carrier.py`, `carrier_field.py` and the three test files I touched | **All checks passed** (was 1 × F821) |

Union over the four runs (the file sets are disjoint): **525 passed, 3 skipped, 1 collateral failure** across 20 test files.  A final
confirming re-run of the three files whose bars I restated
(`test_audit2609_a6_verify_carrier`, `test_audit2609_a6_carrier`,
`test_niche_c5_exact_tilted_reference`) gives **181 passed**.

**The one failure is not WP-A6's and not mine.**
`tests/unit/test_niche_d1_tilted_carrier.py::test_tilted_carrier_beats_the_equivalent_ndarray_wavefront`
asserts, as a fail-before, that the `ndarray`-wavefront branch of
`_compute_carrier` is WORSE than the analytic `TiltedCarrier` one by more than
1e-5 in launch cosine:

```
tests\unit\test_niche_d1_tilted_carrier.py:363: AssertionError
E  assert np.float64(1.7196654544804346e-08) > 1e-05
```

`_compute_carrier` is `lumenairy/elements/_lens_traced.py:4850` -- WP-A3's
file, which `git status` shows mid-edit.  Somebody has made that branch three
decades more accurate, which turns a "must still be worse" pin into a failure.
Nothing in `a18ab074` touches `lumenairy/elements/` (`git diff --stat
a18ab074^..a18ab074 -- lumenairy/elements/` is empty) and nothing in my three
fixes reaches a ray-launch cosine.  Owner: WP-A3 / VERIFY-A3, who should
re-base that pin on the new accuracy or state the claim as a ratio.

No regression anywhere.  In particular the two bars I restated
(`test_audit2609_a6_carrier.py`'s three C4 assertions and
`test_niche_c5_exact_tilted_reference.py`'s re-barred pin) pass in their
scale-free form, and my `_beam_containment_standoff` change moves nothing in
the 292-test batch -- it is provably inert wherever `alpha > 0` or the envelope
is real, and both guard files whose fail-before arms depend on a refusal
(`test_niche_r9_highna_final_leg`, `test_niche_d5_dx_flatness_gate`) sit at
ext < 3.2, where `gamma < 0` keeps the old branch.

---

## 5. Re-check of everything the audit verified correct

All re-run on the fixed tree; every number reproduces:

* **Sziklas-Siegman transform** (`p1_gauss.py`): relL2 **1.064e-07 / 1.994e-07
  / 2.655e-07**, phase RMS 7.803e-08 / 1.571e-07 / 2.156e-07 rad,
  `P_carrier/P_analytic = 1.000000` at m = 1.4865 / 2.9459 / 10.7297 -- the
  audit's and the report's numbers.
* **Converging carrier through focus** (`p1b_focus.py`): peak ratio
  **0.993439**, P = 0.999999 at the focus.
* **Paraxial readout vs the analytic oracle** (`p3_readout.py`): piston-free
  relL2 **1.048e-03** (NA 0.05, ext 4), **1.767e-02** (ext 2, the narrow-grid
  branch), **3.808e-03** (NA 0.10); peak ratio 0.99992 / 0.96849 / 0.99964;
  EE(2 w0) 0.99964 vs 0.99964.
* **`reconstruct`/`envelope` round trip and the aliased-carrier design**
  (`p2_alias.py`): round trip **4.403e-09**, zero warnings at 4.796 rad/px, the
  continued-leg control identical to the field (**7.511e-03** both arms).
* **`R -> inf` guards, `_rereference` no-op, aperture transmission**
  (`p4_fit.py` §4-5): `dx_out/dx = 1.000000000000`, `propagate(R=1e15)` vs
  `R=inf` **4.841e-15**, transmission **0.721850**, `refit_carrier=True`
  returns R unchanged.
* **Standoff-choice invariance** (`p6_invariance.py` §6b): relL2 1.15e-04 at
  200 µm rising to 1.28e-02 at 3 mm, monotone and piston-dominated -- unchanged.
* **A two-group chain against the brute-force ASM + `apply_real_lens_traced`
  arm** (`p5_chain.py`, N = 2048): power ratio **1.000067**, r2m **1.1388170 mm
  vs 1.1438915 mm (0.44 %)**, centroids on axis to **1.143e-10 m** (chain) and
  **3.134e-09 m** (brute) -- the audit's 1.000067 / 0.44 % / 1e-10 exactly.
  (This arm runs `apply_real_lens_traced`, i.e. WP-A3's file, and still
  reproduces.)
* **`carrier_referenced_aperture`**: transmission 0.721850 == the retained
  power fraction; and the `refit_carrier=True` PHYSICAL-FIELD invariance holds
  at every decentre out to 2 waists, flat and curved (relL2 <= 2.2e-16).
* **Backend twins** (`test_niche_k2_carrier_backends.py`): 13 passed, 3 skipped
  (CuPy not installed).  My own JAX checks: complex64/complex128 preserved with
  x64 on and off through `propagate_carrier_referenced`.

---

## 6. Open items for the orchestrator

| ID | Sev | Item |
|---|---|---|
| **OI-1** | **P2** | **C1 is inoperative on grids narrower than `_FOCUS_STANDOFF_MARGIN` = 3.2 beam radii**, which is the whole of the module's own documented small-extent branch.  Measured at ext = 3.0, λ = 0.85 µm, NA 0.08: a 1 % carrier mismatch costs 8.7 % of peak with **zero warnings** (containment 1.627, above the 1.0 floor), and a 3 % mismatch costs 70 % (warned).  *Design*: when `gamma < 0` the target margin `M` is unreachable at ζ = 0, so the beam term should resolve against the best achievable containment rather than returning 0 -- the same trade `_small_extent_focus_standoff_f` already makes on the carrier side.  Concretely: maximise the modelled containment `half(1−ζ/ζ_cf)/w_beam(ζ)` over ζ ∈ [0, |z|] (one more quadratic, closed form) and take the shorter of that leg and the hand-off-error budget the small-extent branch already computes.  ~1 day with the calibration matrix re-run.  Until then the behaviour is an accident of the `standoff > abs(z)` clamp and depends on the sign of `alpha`. |
| **OI-2** | P3 | **The C2 tilt projection has a half-pixel blind spot.**  It is gated on `centre != (0,0)` and `_envelope_amp_centroid` snaps sub-pixel decentres to exactly `(0,0)`, so a beam decentred by < dx/2 while carrying a tilt reads a finite radius (measured **0.255 m at x0 = 0.49 dx, L = 0.02**, truth `inf`; 0.625 m at 0.20 dx), with a discontinuity at the snap.  *Design*: apply `_tilt_free_moment` whenever the CALLER asked for `'auto'` or an explicit centre, independent of where the centroid lands, and move the byte-identity pin to `centre='origin'`.  That is a deliberate default change of a few ulp on centred fields and will need `test_niche_d14_deterministic_carrier_fit`'s pin re-based; ~0.5 day. |
| **OI-3** | P3 | **`_sphere_parab_conversion(dy=)` reaches no caller.**  All 7 call sites pass a single scalar pitch, so a `dy != dx` chain still converts its y axis against `dx`.  The audit asked for "take `dy` or refuse"; the parameter exists and is correct, but the latent defect is unchanged.  *Fix*: thread `dy` from `propagate_carrier_referenced` / `carrier_referenced_reconstruct` / the chain's fine-retrace sites, or refuse a tuple pitch at those sites.  ~2 h. |
| **OI-4** | P3 | **`carrier_field.aggregate`'s return dtype is a silent default change** not called out as such in `WP-A6_CHANGELOG.md`.  An all-complex64 fan now sums in complex64 where it used to sum in complex128: measured **7.6e-08 relL2** on a 16-field sum (the float32 accumulation floor, `sqrt(K)·eps32` = 4.8e-07).  The audit asked for exactly this, so it is not a defect -- but COMMON §8 asks for the migration line, and the changelog only mentions the accumulator's sizing rule.  *Fix*: one sentence in `WP-A6_CHANGELOG.md`. |
| **OI-5** | P3 | **`WP-A6_REPORT.md` should be corrected on two points** before it is filed: §2.3/§5.3's attribution of the surviving complex128 promotion to `propagators/mft.py` (the request should be withdrawn -- the site was `carrier.py:9512`, now fixed), and §2.4's "two decades under the ~1e-11 representation noise" for the separable regrouping (it is AT the floor, and the fixed bar fails above ~3e4 rad).  The C2 summary-table after-numbers for the decentred pure tilt also do not reproduce digit-for-digit (round-off-level, finding unaffected). |

Not defects, recorded for completeness:

* WP-A6's `test_the_asm_axis_band_limit_is_in_register_at_odd_n` describes its
  oracle as "built by hand here on `fftfreq`" but builds it on
  `(arange(n) − n//2)/(n·d)`, i.e. on the expression under test.  The property
  holds against a genuine `fftfreq` oracle (§2, C5), so this is a docstring
  inaccuracy, now covered by my file.
* `CarrierField.__eq__` raises `ValueError` on any comparison (the dataclass
  `__eq__` on an ndarray field).  Pre-existing, not WP-A6's, and out of scope --
  but it means `dataclasses` equality is unusable on this class, which the
  freeze at 5.48 will not change.

---

## 7. Files I touched

* `lumenairy/propagators/carrier.py` -- OI-A (`_beam_containment_standoff`
  root selection + docstring), OI-B (the tilted landing's ramp dtype), the C4
  error-floor derivation, the `_freq_sq_1d_bld` docstring.
* `lumenairy/propagators/carrier_field.py` -- OI-C (`phasor_on`'s lambda
  capture).
* `tests/unit/test_audit2609_a6_carrier.py` -- three bars restated with their
  re-derivation (C4).
* `tests/unit/test_niche_c5_exact_tilted_reference.py` -- the same bar and
  re-derivation on the re-barred byte-identity pin.
* `tests/unit/test_audit2609_a6_verify_carrier.py` -- NEW, 72 tests.
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A6.md`
  -- this report.

No other file was edited; no git write command was run.

---

## 8. Follow-up (2026-09-12) — the five §6 open items, implemented

The orchestrator ruled all five open items in.  All are now implemented and
measured; the file list is at the end of this section.  Nothing else changed.

### OI-1 (P2) — C1 is now operative below 3.2 beam radii of grid

**Root cause.**  Two halves of the same mistake.  (a) The beam-referenced
solver always asked for `_FOCUS_STANDOFF_MARGIN` = 3.2 radii, so on a grid
holding fewer than that at the input plane `gamma = half² − (M·w)²` went
negative, the input plane itself was uncontained, and the solver returned 0.0.
(b) The guard's floor was a bare 1.0 radius, which a landing at 1.627 clears —
so the resulting loss was neither repaired nor reported.

**Fix.**  A new shared helper `_achievable_focus_margin(half, w_env)` returns
`min(M, sat·half/w_env)` — *exactly* the `m_req` the carrier-referenced law in
`_default_focus_standoff` already resolves against, with
`sat = f_cap/√(1+f_cap²)` and `f_cap = √(_FOCUS_STANDOFF_WAIST_GROWTH²−1)`.
Both halves now use it:

* `_beam_containment_standoff` forms `Q = (m_target·w_env)²`, which makes
  `gamma ≥ 0` true by construction, so a leg always exists to resolve;
* `_check_focus_containment` disposes below
  `max(_FOCUS_READOUT_CONTAINMENT_MIN, _FOCUS_READOUT_CONTAINMENT_FRAC·m_target)`
  and publishes `containment_target` beside the two readings.

The helper is the **identity** above `M/sat` = 3.695 radii, so the entire
validated wide-grid surface is untouched by construction, not by luck.

**Measured, λ = 0.85 µm, NA 0.08, ext = 3.0 (target margin 2.598 radii):**

```
R/R0   leg (um) post -> pre   containment post -> pre   peak vs the ORACLE      guard, pre-fix leg
1.00     94.76 ->   94.76        2.745 -> 2.745        0.999324 (control)       silent (contained)
0.99    534.48 ->  167.37        2.598 -> 1.627        0.990892 -> 0.912560     REFUSES
0.97   1408.84 ->  312.71        2.598 -> 0.984        0.954704 -> 0.300483     REFUSES
0.93   2698.45 ->  603.87        2.598 -> 0.865        0.883644 -> 0.025460     REFUSES
```

So the 1 %-mismatch 8.7 % peak loss is **repaired** (0.9126 → 0.9909) *and*
**seen** (its pre-fix leg is now refused where it used to pass silently); the
3 % case goes 0.3005 → 0.9547 and the 7 % case 0.0255 → 0.8836.

**Nothing shipped moved.**  On the resolver's own 6 NA × 10 extent calibration
matrix: **0 cells** whose resolved leg differs from the pre-fix one, **0** guard
firings, worst achieved/target ratio **0.9969** (against the 0.9 bar — 1.108×
of clearance) and worst absolute containment **1.2566**.  `p6c_mismatch.py`
reproduces bit for bit (222.39 / 1008.60 / 1873.95 / 4173.27 / 7144.93 µm,
peaks 1.000000 / 0.999514 / 0.998602 / 0.994304 / 0.985236, 0 warnings), and
the ext = 4.0 fixture is unchanged to every digit (legs 368.89 / 1006.77 /
2051.45 µm, containment 3.200, peaks 0.995468 / 0.973251 / 0.919521).  The
"never shortens" sweep is still **0 shorter over 910 runs**; the count of runs
that lengthen rose from 81 to **184**, which is the narrow-grid gain.

**Bar derivation (two-sided).**  `_FOCUS_READOUT_CONTAINMENT_FRAC = 0.9` sits
1.108× under the tightest passing cell of the calibration matrix (0.9969) and
1.44× / 2.37× / 2.70× over the failing ratios (0.626 at 1 %, 0.379 at 3 %,
0.333 at 7 %).  Like the 1.0 floor beside it that is not decades of clearance,
and the constant says so: it refuses a leg that demonstrably lost the beam, it
does not grade quality.

**Scope — and a false positive of mine that WP-A3 caught.**  My first cut
applied the relative arm on EVERY grid.  WP-A3 reported
`test_niche_d1_tilted_carrier.py` going from 33 passed to 1 failed + 6 errors,
all `_check_focus_containment` refusals at a measured containment of 1.033
against my floor of 2.880.  I measured that fixture before choosing a side, as
instructed:

* its input grid is **4.64 beam radii**, i.e. WIDE — `m_target` is the full
  3.2, so this was never the narrow-grid case OI-1 is about;
* the refused readout is `dx_out = 400 nm, N_out = 1024` — a **409.6 µm
  diagnostic window** holding EE(3 µm) = 0.0014, i.e. the D1 ghost/skirt
  readout, not a focal spot.  Its measured `sqrt(2<r²>)` = 59.76 µm against a
  grid half-width of 61.75 µm reads the WINDOW; the beam's own modelled ABCD
  width there is **17.61 µm** (containment 3.506), so the core is not clipped;
* with the historical floor restored (`_FOCUS_READOUT_CONTAINMENT_FRAC = 0`,
  via a session plugin) the file is **33 passed** — including
  `test_tilted_relay_lands_on_the_exact_ray_trace`,
  `test_tilted_relay_reaches_the_on_axis_diffraction_limit` and
  `test_energy_is_conserved_through_the_tilted_relay`.  The landing at 1.033 is
  accurate to the fixture's own oracles;
* and the leg sweep on that call is incoherent (peak ratios 0.630 / 2.367 /
  4.567 / 3.171 / 5.029 at 0.5x to 5x the resolved leg, relL2 ~1.2 between
  arms), because the window is deliberately wider than one Bluestein period —
  there is no "longer leg is better" truth to move towards.

So the floor was a **false positive**, and the cause was my derivation: I
calibrated FRAC on flat-envelope cells (where achieved ≈ target by
construction) and on the narrow-grid mismatch fixture, then applied it
everywhere.  On a wide grid with a structured envelope the measured second
moment is a poor clipping proxy.  The arm is **scoped to
`m_target < _FOCUS_STANDOFF_MARGIN`** — exactly the grids the absolute floor is
blind on.  Above the knee (`M/sat` = 3.695 radii) the historical 1.0 floor
stands unchanged, so d1 is back to **33 passed with no edit to that file**
(I declined the grant to touch it).

**A second false positive, from the same over-reach.**  That scoping fixed d1
but left `test_niche_tight_focus_readout.py` failing 2 of 15: a landing at
**2.123 radii measured AND 2.123 modelled** (the two agree to 0.1 %, so nothing
is clipped or saturating) on a 3.00-radius grid whose nominal target is 2.598 —
while the guard's own message said *"No leg length reaches even the
2.598-radius margin this grid can give"*.  The bar sat above what the resolver
could deliver for that beam: `m_target` is a property of the GRID, but
reachability also depends on the beam's own curvature and divergence.  So the
relative arm is additionally gated on **a qualifying leg demonstrably
existing** — `_beam_containment_standoff(...) > 0`, computed before the floor is
chosen and reused for the message's actionable remedy.

The arm therefore governs exactly the cells where the absolute floor is blind
AND a better leg exists: `m_target < _FOCUS_STANDOFF_MARGIN and _need > 0`.
`test_niche_tight_focus_readout.py` is back to **15 passed**, and every OI-1
acceptance number above is unchanged by both gates — re-measured after them:
matrix 0 moved / 0 fired / worst ratio 0.9969 / worst absolute 1.2566; the
ext = 3.0 rows identical to the digit; ext = 4.0 identical.  Pinned in
`::test_the_relative_floor_is_scoped_to_grids_below_the_knee`.

**One test of mine genuinely had to be re-based, and that is a RESULT rather
than bookkeeping.**  `test_niche_r9_highna_final_leg.py::test_r9_exact_leg_focuses_highna_sphere`
asserted `ee_par < 0.10` — "the paraxial carrier cannot focus this leg" — and,
per WP-A6's addition, that the default disposition REFUSES that arm.  Both were
true of the old resolver and are false of the new one: those fixtures hold
2.56 / 2.19 beam radii, so the beam term used to return 0.0 and the arm ran on a
leg sized purely from the carrier.  Measured now:

```
NA      leg (um)               containment      EE(2 w0)          FWHM (um)
0.300     8.0250 -> 209.7156   1.055 -> 2.213   0.0033 -> 0.5446  3.996 (exact 1.911)
0.455     3.4965 -> 270.7493   0.919 -> 1.900   0.0019 -> 0.2662  3.326 (exact 1.491)
```

i.e. the C1 fix improves the paraxial high-NA readout by **165x / 140x in
encircled energy** — which is exactly what C1 is for on a fixture whose carrier
badly misdescribes the beam.  The exact arm is untouched (EE 0.9999 / 0.9979,
FWHM 1.911 / 1.491 µm).  The test's subject survives, so its claim is restated
as the COMPARISON it was always about, with derived two-sided bars:
`ee_ex > 1.5·ee_par` (measured 1.836 / 3.748; 1.0 is the null) and
`fwhm_par > 1.8·fwhm_ex` (measured 2.091 / 2.231; 1.0 is a resolved paraxial
arm).  The containment waiver and the "default refuses" corollary are removed,
with the before/after table in the comment.  **9 passed.**

Pinned in `TestVerifyC1AgainstTheAnalyticFocus`:
`test_a_narrow_grid_resolves_and_the_guard_sees_the_pre_fix_leg` (3 params,
each with its own fail-before arm),
`test_the_contained_control_on_the_same_narrow_grid_is_untouched`,
`test_the_achievable_margin_leaves_every_wide_grid_alone`,
`test_the_relative_floor_is_scoped_to_grids_below_the_knee`,
`test_the_sixty_cell_matrix_is_untouched_and_silent`.

### OI-2 — the half-pixel blind spot is gone

`_fit_carrier_inv` gains `project_tilt` (`None` = follow `centre`, the
historical coupling), and `carrier_referenced_fit_radius` passes `True` for
every `centre` but `'origin'`.  The projection is now a decision about what the
CALLER asked for, not about where the snapped centroid happened to land.

**Measured, L = 0.02 on a 100 µm waist at dx = 2 µm, truth `inf`:**

```
x0/dx      0.00       0.20       0.49       0.51       1.00       25.0
before  -4.9e+14 m   0.625 m    0.255 m   2.5e+14 m  8.7e+13 m  -1.8e+15 m
after   -5.8e+14 m  1.8e+14 m  1.4e+14 m  2.5e+14 m  8.7e+13 m  -1.8e+15 m
```

— the 0.625 m / 0.255 m readings are gone, and the discontinuity at the snap
with them.  Cost to a CENTRED field: **0 ulp** on `increment` (exactly equal at
every radius) and **1–2 ulp** on `gradient`.  `centre='origin'` remains the
byte-identity escape hatch and still reads the finite radius, which the test
asserts against the analytic pre-fix form `1/R = L·x0/(x0² + w²/2)` so the
fixture cannot be mistaken for a benign one.  The decentred-parabola and
non-Gaussian results are unchanged (1.0000000000 at 0.5 / 1.0 / 2.0 waists).

Pinned in `TestVerifyC2::test_a_sub_pixel_decentre_no_longer_hides_a_pure_tilt`
(6 params).  The WP's own `test_the_on_axis_answer_is_byte_identical` was
restated as `test_the_on_axis_answer_moves_by_at_most_two_ulp` with its 4-ulp
derivation — the byte-identity claim it made *was* the defect's own gate.

### OI-3 — `dy=` is wired to all seven call sites

The chain now tracks `cur_dy` beside `cur_dx` and picks up the y component of
an astigmatic leg's `(dx_x, dx_y)` instead of discarding it at the two
`isinstance(cur_dx, tuple)` collapse sites; all seven
`_sphere_parab_conversion` calls pass a `dy` (`dx_fine` on the two retrace
sites, `cur_dy` on the five chain sites).  Bit-identical wherever `dy == dx`,
which is every square leg — the helper's `dy=dx` arm is `np.array_equal` to
`dy=None`, and that is pinned.

An astigmatic leg really does produce two pitches — measured
`(4.160, 4.100) µm` from `propagate_carrier_referenced((50, 80) mm, 2 mm)`,
1.5 % apart — and the screen built on that pair matches a numerically stable
hand-built eikonal to **< 1e-12** while differing from the dx-only one by
> 1e-10.  Pinned in
`TestVerifyC5::test_the_sphere_parabola_conversion_dy_is_wired_to_every_caller`
(a static check that all 7 sites pass a `dy`) and
`::test_an_astigmatic_leg_really_produces_the_two_pitches`.

### OI-4 — the `aggregate` migration line

Added to `WP-A6_CHANGELOG.md` under the C3 entry: an all-complex64 fan is now
accumulated in complex64, measured **7.6e-08 relL2** against the complex128 sum
of the same 16 fields (inside the `√K·eps32` = 4.8e-07 random-walk bound), with
the two ways to keep the old precision spelled out.

### OI-5 — `WP-A6_REPORT.md` corrected

* §5.3 request 3 struck through and marked **WITHDRAWN**: `mft.py` preserves
  complex64 (WP-A5's measurement, and both readouts return complex64 here); the
  site was `carrier.py:9512` and it is fixed.
* §2.4's "two decades under the ~1e-11 rad representation noise" replaced by
  the measured `1.0–1.6 × eps·max|arg|` table and the statement that a fixed
  absolute bar does not hold over the range that section itself quotes.
* The C2 summary-table decentred-tilt after-numbers annotated as round-off
  level, with my re-measurement and the claim that survives (order, not digits).

### Follow-up test runs

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a6_verify_carrier.py` | **83 passed**, 27 s (was 72; +11 for OI-1/2/3) |
| `pytest` verify + the WP's file | **164 passed**, 51 s |
| `pytest` verify + the WP's file + the 12 nearest carrier files | **376 passed** |
| `pytest` the 20-file set (those 14 + `d1`, `r9`, `d3`, `tight_focus_readout`, `r8`, `c1_consolidation`, `k2`) | **532 passed, 3 skipped**, 7 flaky (below), 18:39 |
| `pytest tests/unit/test_niche_d1_tilted_carrier.py` | **33 passed** |
| `pytest tests/unit/test_niche_tight_focus_readout.py` | **15 passed** |
| `pytest tests/unit/test_niche_r9_highna_final_leg.py` | **9 passed** |
| `ruff check` on both modules and the three test files | **All checks passed** |
| `repro/CARRIER/p6c_mismatch.py` | bit-identical to §2 |

### The 7 flaky failures in the 20-file run, and why they are not mine

All 7 were in `test_niche_d14_deterministic_carrier_fit.py`, and none is a
containment or fit failure: each is a SPAWNED CHILD that could not
`import lumenairy`, dying at

```
File "...\lumenairy\elements\__init__.py", line 64, in <module>
    from .elements import (
```

i.e. inside another agent's module, in a file neither WP-A6 nor I touch.  That
file's tests compare hashes across child processes, so a module being written
mid-run by a concurrent agent breaks the import in the child while the parent
holds the already-imported version.  WP-A6's own report records the identical
pattern on the identical file (its �4.2: "7 failures ... that did not reproduce
on any later run ... I hit a live `SyntaxError` in `_cache_registry.py` and an
`AttributeError` from `elements/pmm/twod.py` in the same window").

Re-measured: **18 passed alone**, and **98 passed twice consecutively** on the
exact pairing that had failed 2 (`test_audit2609_a6_carrier.py` +
`test_niche_d14_deterministic_carrier_fit.py`); `python -c "import lumenairy"`
is clean.  Not reproducible, and not attributable to this package.

### `na_exit_guard` (VERIFY-A3's note) � measured, and NO change made

VERIFY-A3 added `na_exit_entrance_disc` / `na_exit_output_disc` /
`na_exit_guard` to `_exit_na_out` and asked whether `on_tilt_exact_grid` should
read the conservative one (1.72x larger on a thick f/1.1).

**It should not, and it already does not fire on an NA at all.**  Read at
`carrier.py:7821`, that guard disposes on
`power_frac_above_nyquist > _TILT_EXACT_NA_POWER_TOL` � the *discarded exit
power*, measured by the element on the very grid it just used.  `na_exit`
appears only in the message text, as the number being reported.  The code's own
comment records why the NA was deliberately rejected as the criterion: the
measured exit NA is the marginal ray at the e^-4 AMPLITUDE contour, carrying
~3e-4 of the power, and a 12288-vs-16384 convergence check showed the grid was
adequate (identical FWHM / EE3 / EE6 / EE12) on a leg the NA test called
under-sampled.  Re-pointing it at a 1.72x LARGER NA would reinstate exactly the
false positive that comment documents having removed � and would do so by
refusing legs, not by refining them.  `na_exit` itself is unchanged (VERIFY-A3
confirms it is still the entrance-disc statistic), so the message stays correct
as written.  No change, nothing re-calibrated, and the sizing path
(`dx_fine = lambda/(3 na_exit)`) is untouched.

### Files touched in the follow-up

* `lumenairy/propagators/carrier.py` — `_achievable_focus_margin` and
  `_FOCUS_READOUT_CONTAINMENT_FRAC` (new); `_beam_containment_standoff` targets
  the achievable margin; `_check_focus_containment` gains the relative floor,
  `containment_target` and a re-worded message; the `on_focus_containment`
  docstring; `_fit_carrier_inv` gains `project_tilt`;
  `carrier_referenced_fit_radius` passes it and re-documents `centre`;
  `cur_dy` tracked through the chain and `dy=` on all 7 conversion call sites.
* `tests/unit/test_audit2609_a6_verify_carrier.py` — 11 new tests (83 total).
* `tests/unit/test_audit2609_a6_carrier.py` — the on-axis byte-identity test
  restated as a 4-ulp claim with its derivation.
* `docs/.../fixes/WP-A6_CHANGELOG.md` — OI-4.
* `docs/.../fixes/WP-A6_REPORT.md` — OI-5.
* `docs/.../fixes/VERIFY_WP-A6.md` — this section.

No other file was edited; no git write command was run.

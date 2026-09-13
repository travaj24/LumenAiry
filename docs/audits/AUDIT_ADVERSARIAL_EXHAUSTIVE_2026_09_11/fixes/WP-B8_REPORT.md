# WP-B8 (Wave 4) -- analysis and sources: the performance designs WP-A7 and WP-A11 deferred

Repository `Lumenairy`, branch `audit-fixes-2026-09`, on top of `284daccc`.
Scope: the three A6 rows `WP-A7_REPORT.md` §6 deferred, the three Z3 rows
`WP-A11_REPORT.md` §6 deferred, and the MFT-based PSF sampler of audit
**§15.9**.  Changelog text: `WP-B8_CHANGELOG.md`.

Every measurement in this report was taken on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at
a time, on 2026-09-13.  Python 3.14.6 / NumPy 2.4.6 / SciPy 1.17.1, Windows.

---

## 1. Summary

| # | item | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|---|
| 1a | `compute_psf` transient (A6.1) | **done** | `analysis/psf_mtf_otf.py:58,84,142,151,422,510` | 130 | the explicit-shift expression restated in-process; `array_equal` on raw bytes | peak **4.00 -> 2.00** padded grids (268.4 -> 134.2 MB at oversample 4 on a 512 pupil); 1.06-1.32x time; **byte-identical** |
| 1b | `compute_psf(method='mft')` (§15.9) | **done** | `analysis/psf_mtf_otf.py:151,439` | 17 | brute-force centred Fourier sum (7.7e-15), closed-form Gaussian PSF (1.1e-15), analytic Airy (8.3e-05, pixelation limited) | a 64x64 window at 16x zoom: **2147.5 MB / 6520.6 ms -> 24.5 MB / 18.6 ms** (87.7x / 349.7x); at 32x, 350.9x / 1223.6x.  Default unchanged |
| 2 | `encircled_energy_profile` (A6.2) | **done** | `analysis/psf_mtf_otf.py:741,834,880,1042` | 8 | the unshared pair, bit-for-bit; the profile's own pixel-shell width | a caller wanting both: **1016.0 -> 474.4 ms at N = 2048** (2.14x), 221.7 -> 105.0 at N = 1024; curve and radius **byte-identical** |
| 3a | Zernike basis build (A6.3) | **done** | `analysis/zernike.py:97,240,435` | 7 | the per-mode loop restated in-process | **4426.4 -> 1952.3 ms** at N = 1024 / 66 modes (2.27x; 1.61-2.27x over 9 points), **bit-identical**; peak +18 % to +29 %, bounded by `2/(n_max+2)` of the basis |
| 3b | Zernike recurrence (A6.3) | **done** | `analysis/zernike.py:94,115,153` | 15 | exact rational (`fractions.Fraction`) evaluation, no library, no mpmath | the factorial sum's own error: 0.0 (n<=6), 1.5e-09 at n=22, 3.2e-03 at n=40; the recurrence <= 3.9e-15 to n=32.  Switch at **n >= 22**; n < 22 unchanged bit for bit |
| 3c | DM influence functions (A6.3) | **done** | `analysis/ao.py:99,171,231,249,290,375` | 6 | the materialised design matrix; the per-actuator loop restated | `_banded_IF_apply` hoisted; three dense meshgrids -> broadcast views (bit-identical); a 512 MiB eager stack now warns once.  **No number moves** |
| 4 | Gori pseudo-modes (A11 §6.1) | **done** | `sources/core.py:2084,2094,2117,2135,2183,2456,2646` | 14 | the exact finite-M moment `E[I^2]/E[I]^2 = 2 - 1/M`; chi-square vs `Exp(1)`; the analytic Gaussian kernel | per realisation at N = 512: **154.14 -> 11.43 ms** and 125.85 -> 20.99 MB (**13.5x / 6.0x**), i.e. the design's ~100x penalty estimate inverted.  Default `'fft'` byte-identical |
| 5a | `create_gaussian_beam(geometry_dtype=)` (A11 §6.2) | **done** | `sources/core.py:384` | 32 | the float64-geometry field | opt-in float32 geometry: peak **67.1 -> 50.4 MB** (2.00x -> 1.50x the output) and 93.6 -> 55.7 ms at N = 2048, tolerance **1.19e-07** of the peak.  Default byte-identical |
| 5b | `apply_jones_matrix` accumulation (A11 §6.3) | **done, bit-identical** | `elements/polarization.py:633,722` | 27 | the pre-fix expression restated in-process | **268.4 MB (4.00 grids) / 131.8 ms -> 201.3 MB (3.00) / 96.4 ms** at N = 2048; 3.00 is the floor, not 2.00 |

256 new tests, all green.  1125 recorded results from the pre-change library
re-run in a separate process: **1125 byte-identical**.

---

## 2. Per item

### 2.1 `compute_psf` memory (A6.1) -- and why NOT the chessboard identity

The A7 design named the separable chessboard identity `chess * fft2(chess *
a)`, exact for even N.  It is exact **mathematically**; the two forms are
different floating-point reductions, and that is the question the item turns
on, because COMMON requires bit-identity or a documented tolerance and this
is a default path.

Measured over N = 2...1024 on three input families (complex noise, an
aberrated pupil, a single hot pixel), the chessboard form is bit-identical to
`fftshift(fft2(ifftshift(a)))` **exactly when both lengths are a power of
two**:

| N class | example | agreement |
|---|---|---|
| power of two | 2, 4, ..., 1024 | **bitwise identical** |
| even, not a power of two | 100, 192, 384 | 1.3e-16 ... 9.0e-16 relative -- a moved default |
| odd | 17, 63, 129 | relative 1.5-2.0 -- a different array (the cyclic shift) |

A power-of-two PSF grid is the common case, but "bit-identical on this
build's pocketfft radix choice" is not a property anyone can rely on, and the
even-non-power-of-two arm would move `compute_psf`'s output silently.

What shipped instead is the **in-place quadrant exchange**.  For an even axis
`fftshift` and `ifftshift` are the same roll by `N // 2`, so in two dimensions
both are one quadrant exchange, which is a PERMUTATION of the values: the FFT
is handed a buffer holding bit-for-bit what `ifftshift` would have put in a
fresh copy, and the second exchange is `fftshift` by definition.  Bit-identity
is a construction property, not a measurement.  It also covers every even
length rather than only the powers of two.

Two further pieces get the count from 4 to 2 rather than 3:

* `fft2` is written out as `fft(axis=-1)` then `fft(axis=-2)` -- which is what
  `_raw_fftnd` does, it walks the axis list in reverse (pinned) -- so the
  padded input can be released between the passes;
* ownership of the padded array is transferred through a one-element box
  (`_centred_fft2_take`), because Python has no move and a plain argument
  leaves the caller's name holding a whole padded grid across the transform.

`compute_psf` additionally reads the pupil-side half of the `'power'` Parseval
ratio BEFORE the transform (same expression, same array -- only the moment
moves), builds `|amp|^2` through one real buffer and applies both
normalisations in place.

Interleaved medians, against the pre-fix expression restated in the same
process:

| N_pupil | ovs | N_psf | padded grid | before | after | time |
|---|---|---|---|---|---|---|
| 512 | 2 | 1024 | 16.8 MB | 67.1 MB (4.00x) | 33.6 MB (2.00x) | 86.5 -> 65.6 ms |
| 512 | 4 | 2048 | 67.1 MB | 268.4 MB (4.00x) | 134.2 MB (2.00x) | 368.2 -> 347.3 ms |
| 1024 | 2 | 2048 | 67.1 MB | 268.4 MB (4.00x) | 134.2 MB (2.00x) | 362.4 -> 298.5 ms |

**`compute_otf`'s peak does not move, and the report says so.**  Its input is
a real intensity PSF, and NumPy makes its own complex copy of a real input for
the first pass; that floor (2.50 grids) sits below what either form reaches.
What the rewrite removes there is the `fftshift` copy and the `/ dc` copy,
neither of which is at the peak.  The test asserts the honest claim -- the
peak does not rise, and the output is bit-identical -- rather than a
before/after that is not there.

### 2.2 MFT-based PSF sampling (§15.9)

`compute_psf(..., method='mft', dx_psf=...)` routes the Fraunhofer integral
through `fraunhofer_propagate_mft` (consumed, not modified).  `'fft'` stays
the default.

**The compatibility statement.**  With `dx_psf=None` the MFT samples exactly
the lattice the padded FFT delivers: `alpha = dx_pupil*dx_psf/(lambda f) =
1/N_psf`, and for even `Np` and `N_psf` the pad offset `(N_psf - Np)/2` makes
the MFT's `n - Np/2` centring and the FFT's `n - N_psf/2` the same phase.
Measured agreement over N_pupil 64/128/256 x oversample 1/2/4 x three
`normalize` modes: **worst 1.1e-15 of the peak**.

On an ODD length they differ by half a pixel and nothing else: `ifftshift`
centres an odd axis on index `N // 2`, the MFT follows the package
convention `(arange(N) - N/2)*dx`, and for odd `N` those are not the same
sample (measured relative 0.77 at Np = 129).  This is a PRE-EXISTING
inconsistency between the module's implemented grid and the package's
documented one; moving either would move a default, so `method='mft'` warns
and names the half pixel.  Recorded for the coordinator in §6.

**Normalisation.**  `|E_out|^2` from `fraunhofer_propagate_mft` already
carries `|prefactor|^2 = (dx_pupil^2/(lambda f))^2`, and that constant IS the
Parseval ratio the FFT path measures empirically on the full grid (both equal
`dx_pupil^4/(lambda f)^2`).  So `'power'` needs no rescale on the MFT path,
and must not have one: on a zoomed sub-field an in-window rescale would force
the visible fraction of the energy to equal the whole pupil's.  Pinned: an
8x8 window at 8x zoom holds under half the energy, and its intensity at a
given physical point matches a 256x256 window's at the same point to 1e-12.
`'none'` divides the constant back out to recover the FFT convention.

**Oracles** (none produced by this library):

| oracle | what it bounds | measured |
|---|---|---|
| brute-force centred Fourier sum at the MFT's own output coordinates | the sampler itself, exactly | 7.7e-15 / 1.9e-15 / 1.3e-15 at 1x / 4x / 11x zoom |
| closed-form Gaussian PSF, on a grid where aliasing `exp(-pi^2 w^2/dx^2) = 4.7e-275` and truncation `exp(-2(L/2)^2/w^2) = 4.4e-223` | the physics | 1.1e-15 / 1.6e-15 / 2.2e-15 at 1x / 4x / 16x |
| analytic Airy pattern of a circular pupil | the physics, pixelation limited | 8.3e-05 at 300 px/diameter, 7.4e-05 at 600; **the same at 1x and 8x zoom**, which is the point -- the residual is the sampled aperture's, not the sampler's |

**Cost.**  Measured on a 512 pupil delivering a 64x64 window, warm:

| zoom | N_psf the FFT needs | fft peak | fft ms | mft peak | mft ms |
|---|---|---|---|---|---|
| 1 | 512 | 8.4 MB | 20.9 | 24.5 MB | 20.2 |
| 2 | 1024 | 33.6 MB | 70.6 | 24.5 MB | 20.4 |
| 4 | 2048 | 134.2 MB | 332.7 | 24.5 MB | 23.0 |
| 8 | 4096 | 536.9 MB | 1462.7 | 24.5 MB | 23.2 |
| 16 | 8192 | 2147.5 MB | 6520.6 | 24.5 MB | 18.6 |
| 32 | 16384 | 8589.9 MB | 27797.0 | 24.5 MB | 22.7 |

and on the FULL natural grid the ranking reverses (the Bluestein pads to
`N_pupil + N_psf`): 3.1x / 4.1x / 7.5x the memory and 2.5x / 3.8x / 5.0x the
time at oversample 4 / 2 / 1.  **The A7 design's "~0 extra memory" does not
hold for lumenairy's Bluestein-based MFT** -- Soummer's matrix triple product
has that property, the chirp-Z does not.  What IS true, and is the reason to
ship it, is that the MFT's cost is flat in the zoom.

`centre_out` (an off-axis window, the coronagraph case) was deliberately not
exposed: `compute_psf` returns only `(psf, dx_psf)`, so a caller could not
tell where an off-axis window sat.  Recorded as deferred (§7).

### 2.3 `encircled_energy_profile` (A6.2)

`encircled_energy_profile(E, dx, *, dy=None, centroid=None) -> (r_sorted,
p_cum, r_max)` is the construction `_ee_sorted_cumulative` already was,
promoted to a public value, and both consumers take it as `profile=`.

| N | two calls, unshared | shared profile | |
|---|---|---|---|
| 1024 | 221.7 ms | 105.0 ms | 2.11x |
| 2048 | 1016.0 ms | 474.4 ms | 2.14x |

Peak memory is unchanged (75.5 MB and 302.0 MB in both arms): the profile is
the same two arrays either way.  Curve and radius come back **bit-identical**
to the unshared path.

Three details worth recording:

* **`p_cum` is clamped in place only when the function built it.**
  `encircled_energy_radius` clamps `p_cum` into `[0, 1]`; doing that through a
  caller's profile would corrupt an array they are about to hand to the
  curve.  Pinned.
* **A contradicting `dy=` / `centroid=` is refused.**  Both are frozen into
  the profile, so accepting one alongside it would quietly answer a different
  question.
* **"The radius is the exact inverse of the curve" is pinned with the bar the
  data supplies.**  The round trip closes to within ONE pixel-radius shell --
  the forward sampler brackets a tie block with `searchsorted(..., 'right')`
  and the inverse with `'left'`, so they pick opposite ends of the riser, and
  a square grid produces those ties constantly.  The test computes that
  riser from the profile at each threshold (measured errors 6.8e-07 ... 1.7e-03
  against risers of 4.3e-04 ... 8.4e-03) and demonstrates the bar is
  separating a real signal: the same round trip through a profile centred
  15 um off misses by **0.55**, 66x the largest riser.

**No cache** (audit §15.5).  A content key would hash ~67 MB per call at
N = 2048; anything less than the content is the incomplete-key shape §15.5
names.  The test asserts the profile holds no reference to the field, is not
memoised between equal-but-distinct arrays, and does not alias its input.

### 2.4 The Zernike recurrence and the DM cache (A6.3)

**The performance half is not the recurrence.**  `_zernike_radial`'s cost is
`np.power(rho, k)` -- a libm `pow` per pixel -- and the first 21 modes issue
34 of them over 6 distinct exponents.  Sharing a `{k: rho ** k}` memo and the
`rho <= 1` mask across the modes is **1.61-2.27x** (9 points, N = 512...2048 x
21/36/66 modes) and is **bit-identical** at every one of them.  The
recurrence saves a further 1.04-1.56x on a single high-n evaluation, which is
not where the time is.

Two traps this item could have fallen into, both caught by the bit-identity
test rather than by reading:

* `N * R * angular` associates LEFT.  Accumulating as `(R * angular) * N` is
  the "same" expression and is measurably NOT bit-identical.
* `R = R + t` with `R = np.zeros_like(rho)` cannot become `R = t` for the
  first term: they differ when `t` is `-0.0`.  The zero start is kept.

The memo needs no budget knob, and that is a derived statement: it holds
`n_max + 1` columns against the basis's `n_modes`, i.e. `2 / (n_max + 2)` of
the array the function is already committed to returning -- 0.33 / 0.29 /
0.22 / 0.17 at 15 / 21 / 36 / 66 modes.  Measured peak cost +29 % at N = 1024
/ 21 modes and +18 % at 66, matching 6/21 and 11/66.  It is a time-for-memory
trade and the changelog says so.

**The accuracy half is a real finding.**  The factorial sum alternates
factorials of `n` and loses about one decimal digit every two orders.
Measured against an exact rational oracle -- `fractions.Fraction` on
`rho = k/128`, exact because the radial polynomial has integer coefficients,
needing neither mpmath nor the library:

| n | 0-6 | 8 | 14 | 20 | 21 | 22 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| factorial sum | 0.0 | 7.1e-15 | 1.8e-12 | 5.0e-10 | 8.9e-10 | **1.5e-09** | 7.3e-09 | 7.0e-06 | 3.2e-03 |
| Kintner | 0.0 | 1.1e-16 | 1.7e-15 | 1.9e-15 | 2.1e-15 | 2.0e-15 | 3.0e-15 | 2.8e-15 | 2.4e-15 |

`_ZERNIKE_RECURRENCE_MIN_N = 22` is where the sum first passes 1e-9.  Below
it nothing moves -- `j < 253` covers every mode the module names, every
docstring example and every realistic decomposition -- and the test asserts
both sides of the boundary: bit-identity with the sum at `n = 21`, and at
`n = 22` that the new answer beats the old against the oracle by more than
three decades.  Stability limit stated: <= 3.9e-15 at every `(n, m)` with
`n <= 32`, <= 3.0e-15 to `n = 40`.

**The DM.**  `_banded_IF_apply` is the hoist A7 asked for, verified against an
explicitly materialised design matrix to 1e-15 relative (bar derived as
`4 N^2 eps`, the blocking floor of the same reduction).  The three dense
`np.meshgrid` sites become broadcast views (S3-7), bit-identical.

The warning is where the honest answer differs from the design.  A7 asked for
`cache_basis` to "default to False above a byte budget"; `'auto'` **already**
does that, and the audit's own case -- a 16x16 DM on 512x512 -- is
**exactly 536 870 912 bytes, i.e. exactly the inclusive `'auto'` ceiling**, so
it caches half a gigabyte in silence.  Moving the boundary was rejected: the
cached `phase()` is one `einsum` over the stack and the lazy one accumulates
actuator by actuator, which are different summation orders, so the boundary
IS a numerical default.  What shipped is a one-shot construction warning above
half the ceiling, naming the bytes and the `cache_basis=False` escape and
saying that the two paths sum differently.  No number moves; both `phase()`
paths are pinned against a local restatement.

### 2.5 Gori pseudo-modes (A11 §6.1)

`generator='fft' | 'modes'` on `_schell_phase_realizations`, forwarded by
`create_gaussian_schell_source` and `create_schell_model_source`.  Default
`'fft'`, byte-identical (asserted both as "naming the default == saying
nothing" and in the 1125-result byte-identity matrix).

The construction and why it is exact are in the changelog.  Three results
worth separating out:

**The WP-A11 cost estimate is wrong by three orders, in the favourable
direction.**  It priced the pseudo-mode sum at ~100x the padded FFT at
M = 256, N = 512.  Measured per realisation at `sigma_g = L/8` with the
heuristic M = 128:

| N | fft ms | modes ms | fft peak | modes peak |
|---|---|---|---|---|
| 64 | 2.16 | 0.81 | 1.98 MB | 0.86 MB |
| 128 | 8.85 | 2.03 | 7.88 MB | 2.11 MB |
| 256 | 37.89 | 4.26 | 31.47 MB | 6.30 MB |
| 512 | 154.14 | 11.43 | 125.85 MB | 20.99 MB |

2.6x to 13.5x cheaper in time, 2.3x to 6.0x in peak.  The estimate priced the
FFT at `N`; the Z2 anti-wrap pad actually runs it at 2-5x `N` per axis, so the
padded FFT runs on 4-16x the area the estimate assumed (4x at
`sigma_g = L/8`, up to the 16x the `_SCHELL_MAX_PAD_GROWTH` cap allows).  Cost crossover on a 512
grid, at fixed N: 10.1 / 18.7 / 37.9 / 69.3 ms at M = 128 / 256 / 512 / 1024
against the FFT's 154, i.e. near **M = 2300**.

**The `M` heuristic and both clamps are derived.**
`M = clip((Lx/sigma_g)(Ly/sigma_g), 128, 4096)` -- the coherence-cell census,
which is the number of independent `1/L`-wide k-space cells inside the
`1/sigma_g` support.  The lower clamp comes from the exact finite-M moment of
a random-phasor sum, `E[I^2]/E[I]^2 = 2 - 1/M` against 2 for a circular
Gaussian: M modes leave a contrast error of exactly `1/M`, and 128 puts it
under 1 %.  The upper clamp is the cost cap, and its warning says precisely
what it costs -- the k-space resolution of a single realisation -- and what it
does not: the ensemble kernel is exact at any `M >= 1`, because the mode
directions are redrawn per realisation.

**Verification.**  (a) The realised correlation lands inside the sampling
error of the Gaussian target for BOTH generators at `sigma_g = L/3` and
`L/8`, with the bar derived as 6 standard errors of
`1/sqrt(n_real * cells)`.  (b) A chi-square of `|phi|^2` against `Exp(1)` on
20 equiprobable bins passes at `M >= 128` (43.82 = the 0.1 % critical value at
19 d.o.f.), sampling one pixel per `4 sigma` cell so the samples are
independent; the `'fft'` generator is put through the same estimator as a
control, so the test is not rigged.  (c) The exact `2 - 1/M` moment is matched
within 5 standard errors at M = 8, 32, 128 and 512 -- an oracle with a closed
form at every M, not an asymptotic one.  (d) The edge-to-edge correlation
reads < 0.05 against a true 3.4e-14, where the pre-Z2 periodised path
reproduced in the same process (`generator='fft', pad_sigma=0.0`) reads > 0.5.

`pad_sigma` with `generator='modes'` raises rather than being ignored: there
is no transform to wrap, so `pad_sigma=0.0` does NOT reproduce the pre-Z2
kernel there, and silently accepting it would be a trap.

### 2.6 `create_gaussian_beam(geometry_dtype=)` (A11 §6.2)

Opt-in, default byte-identical over 36 configurations.  At N = 2048 /
complex64: peak 67.1 -> 50.4 MB (2.00x -> 1.50x the 33.6 MB output), 93.6 ->
55.7 ms.  Documented tolerance, measured over N in {64, 512, 2048} x three
`normalize` modes x on- and off-axis centres: **1.19e-07 of the peak**, one
float32 ULP of the exponent.

`geometry_dtype=np.float32` with a `complex128` output **raises**.  A
double-precision request filled from a single-precision exponent is a silent
precision trap; the error message says so and names the fix.  The returned
`x` / `y` axes stay float64 -- they are the caller's coordinates.

### 2.7 `apply_jones_matrix` (A11 §6.3)

**Bit-identical, and the item is therefore done rather than left.**  Measured
at N = 2048: 268.4 MB (4.00 full-grid complex arrays) / 131.8 ms -> 201.3 MB
(3.00) / 96.4 ms.

**3.00 is the floor, not 2.00.**  The audit's "halves it" is not reachable:
both results must exist at the end, none of the four products can be written
into a result before the other term of that result exists, and `field.Ex` /
`field.Ey` belong to the caller until the last product is read.  Two outputs
plus one scratch is the minimum.

The bit-identity gate is the interesting part.  `a += b` computes in
`result_type(a, b)` and then NARROWS to `a`'s dtype -- a different answer from
`a + b` whenever the two differ, which is exactly a mixed-precision
`JonesField` (`Ex` complex64, `Ey` complex128; the A11 suite already pins one).
Each in-place step is therefore conditioned on the dtypes already agreeing,
and the mixed case keeps the original expressions.  The complex-multiply
operand order is preserved everywhere, per A11's measured 1.8e-15
non-commutativity.  Pinned over 3 sizes x 2 dtypes x {plain, dark, NaN/inf,
1e-160} x {array, spatial callable} and the mixed-precision fall-back.

---

## 3. Files touched

**Modified (all inside my ownership):**

- `lumenairy/analysis/psf_mtf_otf.py` -- `_swap_halves_inplace`,
  `_centred_fft2_take`, `_centred_fft2`, `_scaled`, `_compute_psf_mft`,
  `encircled_energy_profile`, `_resolve_ee_profile`; `compute_psf` gains
  `method=` / `dx_psf=`; `compute_otf`, `encircled_energy_curve`,
  `encircled_energy_radius` bodies; `__all__`.
- `lumenairy/analysis/zernike.py` -- `_ZERNIKE_RECURRENCE_MIN_N`,
  `_rho_pow`, `_zernike_radial_kintner`, `_zernike_polynomial_core`;
  `_zernike_radial` and `_zernike_basis_matrix_build` bodies;
  `zernike_polynomial` delegates.
- `lumenairy/analysis/ao.py` -- `_IF_CACHE_WARN_BYTES`,
  `DeformableMirror._grid_views`, `._banded_IF_apply`; `__post_init__`,
  `_build_IF_basis`, `fit_phase`, `_influence_function_kth`, `phase`.
- `lumenairy/analysis/__init__.py` -- `encircled_energy_profile` in the
  import block and `__all__`.
- `lumenairy/sources/core.py` -- `_GORI_MIN_MODES`, `_GORI_MAX_MODES`,
  `_gori_mode_count`, `_gori_pseudo_mode_realizations`;
  `_schell_phase_realizations` gains `generator=` / `n_pseudo_modes=`;
  forwarded by `create_gaussian_schell_source` and
  `create_schell_model_source`; `create_gaussian_beam` gains
  `geometry_dtype=`.
- `lumenairy/elements/polarization.py` -- `_jones_mix_2x2`;
  `apply_jones_matrix` body.

**History documents re-recorded in the same change** (all five, with reasons):
`docs/history/lumenairy.analysis.psf_mtf_otf.md`,
`lumenairy.analysis.zernike.md`, `lumenairy.analysis.ao.md`,
`lumenairy.sources.core.md`, `lumenairy.elements.polarization.md`.
`lumenairy.analysis.coherence.md` is untouched -- `analysis/coherence.py`
needed no change; the Schell generator lives in `sources/core.py`.

**Added:** `tests/unit/test_audit2609_b8_analysis_sources.py` (256 tests),
`docs/audits/.../fixes/WP-B8_REPORT.md`, `.../WP-B8_CHANGELOG.md`.

Nothing outside that list was touched.  In particular `propagators/mft.py` and
`_bluestein.py` are CONSUMED unchanged, and no file belonging to a concurrent
Wave-4 engineer (lens, carrier, propagator kernels, rcwa/eme/bor, pmm,
raytrace) was opened for writing.

---

## 4. Tests run

| command | result | time |
|---|---|---|
| `pytest tests/unit/test_audit2609_b8_analysis_sources.py` | **256 passed** | 18.4 s |
| `pytest tests/unit/test_audit2609_a7_{detector_sh,image_plane_wfe,misc,opd_unwrap,strehl_reference}.py tests/unit/test_audit2609_verify_a7{,_wfe}.py tests/unit/test_audit2609_a11_polar_sources_infra.py tests/unit/test_audit2609_a15b_reexports.py tests/unit/test_audit2609_a17_history_lint.py` | **266 passed** | 42.9 s |
| `pytest tests/unit -k "psf or mtf or zernike or encircled or ao_ or coherence or schell or gaussian_beam or jones"` | **855 passed, 11 skipped, 1 failed (NOT mine -- see §4.1)** | 450 s |
| `python validation/run_all.py test_analysis test_ao test_coherence test_detector test_features test_sources test_polarization` | **ALL 7 files passed** | 41.8 s |
| `ruff check lumenairy/analysis/ lumenairy/sources/ lumenairy/elements/polarization.py tests/unit/test_audit2609_b8_analysis_sources.py` | **All checks passed** | - |
| `python scripts/record_history_fingerprints.py --check` | **OK on all five of mine** (drift elsewhere is other WPs' -- §4.1) | - |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py` | green on all five of mine | 72 s |
| `pytest tests/unit/test_v4_16_0_walker_all_symmetry.py tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py tests/unit/test_v5_1_0_agent_g_split.py tests/unit/test_s3_7_broadcast_grid.py` | 260 passed, 1 failed -- the `__all__`-symmetry walker, pending §5 | 2.7 s |

### 4.1 Failures that are not mine (measured, not assumed)

* **`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`.**
  Raises `_EnergyError` from `lumenairy/elements/rcwa/_core.py:1058`, a file
  another Wave-4 engineer has modified in the worktree (`git status`:
  `M lumenairy/elements/rcwa/_core.py`, `oned.py`, `twod.py`).  Nothing I
  touched is on that path.  Attribution measured rather than asserted: the
  same `rcwa_jones_1d_segments` ladder the test drives closes to 7.0e-14 /
  4.0e-14 on the pristine `HEAD` library **and** to 6.9e-14 / 3.9e-14 on the
  live worktree, so the failure is in the test's engineered pre-branch-cut
  arm, inside the RCWA package, and belongs to the concurrent WP.
* **`record_history_fingerprints --check` drift on `carrier`,
  `lumenairy.elements._lens_traced`, `lumenairy.raytrace.*`.**  Those files
  are modified in the worktree by other engineers; all five of mine are `OK`.
* **`test_v4_16_0_walker_all_symmetry.py::test_all_submodule_entries_reexported_or_exempt`.**
  Red on my `encircled_energy_profile` AND, independently, on WP-B9's new
  `lumenairy.raytrace.intersection.normalize_directions` and
  `lumenairy.raytrace.surface.is_pure_spherical` (neither exists at `HEAD` --
  checked against the archive).  It needs the top-level re-export in §5, which
  is outside my ownership.

### 4.2 The byte-identity proof

`git archive HEAD lumenairy` extracted READ-ONLY into the scratch directory
is the pre-change library.  (`HEAD` is the right baseline rather than `HEAD^`:
all six files I own are unmodified in the worktree at `284daccc` -- verified
by SHA-256 after CRLF normalisation, `git archive` being the only difference.)

One probe script exercises 1125 named results through APIs that exist in BOTH
revisions -- no `method=`, no `profile=`, no `generator=`, no
`geometry_dtype=` -- and writes each result's RAW BYTES, dtype and shape.  It
is run twice as a child process, once with `cwd` and `PYTHONPATH` set to the
archive and once to the worktree, asserting `lumenairy.__file__` is under the
expected root each time; never through pytest.  The comparison is
`np.array_equal` on the `uint8` views.

Coverage: `compute_psf` (4 pupil sizes including odd x 4 pupil kinds x 3
oversamples x 3 normalisations, plus complex64, real-valued and
Fortran-ordered inputs), `compute_otf` / `compute_mtf` / `mtf_radial` /
`mtf_cutoff`, `encircled_energy_curve` / `_radius` (3 sizes x anisotropic
pitch x 4 field kinds including the delta and the zero field x 3 radius grids
x 5 thresholds x an explicit centroid), `zernike_basis_matrix` /
`_decompose` / `_reconstruct` / `zernike_polynomial` (every `(n, m)` to
n = 12), `DeformableMirror.phase` / `_influence_function_kth` / `fit_phase` /
`apply` on all three `cache_basis` settings, `create_gaussian_beam` (3 sizes x
3 normalisations x 2 dtypes x on- and off-axis), `_schell_phase_realizations`
(both `pad_sigma` arms), all three Schell / incoherent factories, and
`apply_jones_matrix` (array and callable) plus `apply_polarizer` /
`apply_waveplate` / `apply_rotator` / `apply_quarter_wave_plate` /
`stokes_parameters` on 3 sizes x 2 dtypes x {plain, dark, NaN/inf, 1e-160}.

Result, re-run after the last edit: **1125 / 1125 byte-identical.**

---

## 5. Requested changes outside my ownership

**One, in `lumenairy/__init__.py`.**  `encircled_energy_profile` is a new
public name in `lumenairy.analysis.__all__` and belongs at the top level next
to its two siblings, which are both there; the `__all__`-symmetry walker
(`tests/unit/test_v4_16_0_walker_all_symmetry.py`) is red until it lands.  The
exact edit, two lines:

```diff
--- a/lumenairy/__init__.py
+++ b/lumenairy/__init__.py
@@ (the alphabetical analysis import block, currently line 63)
     encircled_energy_curve,
+    encircled_energy_profile,
     encircled_energy_radius,
@@ (the top-level __all__, currently line 1577)
     'encircled_energy_curve',
+    'encircled_energy_profile',
     'encircled_energy_radius',
```

It is the same request WP-A7 made for `unwrap_phase_2d` and
`zernike_basis_cache_bytes`, and the identity assertion belongs in
`tests/unit/test_audit2609_a15b_reexports.py::_REEXPORTS` as
`'encircled_energy_profile': 'lumenairy.analysis.psf_mtf_otf'`.  Neither file
is mine.

**No change is needed in `propagators/mft.py`.**  `fraunhofer_propagate_mft`
and `_bluestein_centred_2d` were consumed exactly as they stand; the only
thing I would have wanted from them -- an `n_centre_in` that matches
`ifftshift`'s `N // 2` convention on an odd grid -- would itself move a
default, so it is documented and warned about instead (§6).

---

## 6. Coordinator follow-ups (observations, no change made)

1. **Odd-N grid centring disagrees between `ifftshift` and the package
   convention.**  `compute_psf(method='fft')` centres an odd pupil on index
   `N // 2`; `(arange(N) - N/2)*dx`, which CONVENTIONS §5 and every source
   factory use, centres it on `N/2`.  For odd `N` those are half a pixel
   apart, which is why the two samplers disagree there (measured relative
   0.77 at Np = 129).  This is the same class as the A11 Z4 finding
   `_plane_wave_carrier` centred its grid on `nx // 2` while the package
   uses `N / 2`, and it is not confined to this module -- anything that
   pairs `fftshift`/`ifftshift` with the package coordinate arrays inherits
   it.  A sweep for the pattern would be worth a work package; fixing it
   inside `compute_psf` alone would move a default.
2. **`_bluestein`'s pad makes the MFT the wrong tool on the natural grid.**
   Soummer's matrix triple product is `O(N^2 M)` with no padding and no
   intermediate larger than `max(N, M)^2`; lumenairy's chirp-Z pads to
   `next_fast_len(N_in + N_out - 1)` and transforms that, which is why
   `method='mft'` costs 3.1-7.5x the padded FFT's memory when both deliver
   the same lattice.  A direct-matrix branch below a size threshold would
   make the MFT strictly better everywhere; it is `propagators/mft.py`, not
   mine.
3. **`DeformableMirror`'s `'auto'` ceiling is inclusive** and the audit's own
   16x16-on-512 case sits exactly on it (536 870 912 bytes).  Flipping `<=`
   to `<` would move the delivered phase map (different summation order), so
   it wants a deliberate decision, not a quiet edit.

---

## 7. Deferred, with designs

1. **An off-axis MFT window (`centre_out`).**  `fraunhofer_propagate_mft`
   already takes it, and it is the reason coronagraph codes want an MFT at
   all.  Not exposed here because `compute_psf` returns only
   `(psf, dx_psf)`: a caller could not tell where an off-axis window sat, so
   exposing it needs the return contract to grow (a `centre_psf` echo, or an
   `(x, y)` pair) -- a signature decision, not a performance one.  *Design:*
   add `centre_psf=(0.0, 0.0)` forwarded to `centre_out`, and return the
   window centre alongside `dx_psf` behind a `return_grid=False` flag so the
   2-tuple default is preserved.  *Effort:* half a day including the
   period-warning interaction.
2. **A direct-matrix MFT branch** -- see §6.2.  Belongs in
   `propagators/mft.py`.
3. **A pseudo-mode generator for the non-Gaussian Schell kernel.**  Gori's
   construction generalises to any kernel that is a characteristic function
   (i.e. any positive-definite `mu`): draw `k_j` from the Fourier transform of
   `mu` instead of from a normal.  `create_schell_model_source` only offers
   the Gaussian `mu` today, so there is nothing to generalise to yet; worth
   recording because the generator makes a non-Gaussian `mu` nearly free
   while the FFT path would need a new filter and a new pad analysis.
4. **`_zernike_radial` above n = 40.**  The recurrence was measured exact to
   the float64 floor out to n = 40 and the literature claims stability to
   n ~ 100; beyond 40 the exact rational oracle gets expensive enough that I
   did not sweep it.  Nothing in the library reaches there (n = 40 is
   j >= 820).  *Design:* extend the oracle sweep with a coarser `rho` grid
   and record the limit in `_ZERNIKE_RECURRENCE_MIN_N`'s block.
5. **The `'auto'` DM ceiling** -- see §6.3.  A decision, not a design.

Deliberately NOT done, and why: the chessboard identity (§2.1 -- it moves the
default on even non-power-of-two grids); moving the DM cache boundary (§2.4 --
it moves the phase map); an in-window `'power'` rescale on the MFT path
(§2.2 -- it is wrong on a sub-field); and `apply_jones_matrix` at 2.00 arrays
(§2.7 -- it is unreachable, not merely undone).

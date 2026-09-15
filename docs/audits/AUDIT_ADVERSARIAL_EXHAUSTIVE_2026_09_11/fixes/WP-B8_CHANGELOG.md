# WP-B8 changelog text -- analysis and sources performance (5.47.0)

The performance designs `WP-A7` §6 (finding **A6**) and `WP-A11` §6 (finding
**Z3**) deferred, plus the MFT-based PSF sampler of
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` **§15.9**.

No default moves.  Every path whose defaults are unchanged is proved
byte-identical against the pre-change library run in a separate process
(1125 recorded results, 1125 byte-identical); see `WP-B8_REPORT.md` §5.

---

### Performance -- analysis/psf: `compute_psf` transient 4.00 -> 2.00 padded grids, byte-identical (A6)

`fftshift(fft2(ifftshift(a)))` materialises the `ifftshift` copy, both
intermediates `fft2` makes internally (it transforms axis -1 then axis -2)
and the `fftshift` copy.  With the padded pupil still alive that is a peak of
**four** full padded complex grids -- 1073.7 MB of transient at oversample 4
on a 1024 pupil, which is the audit's headline number for this function.

Both shifts are now quadrant exchanges done **in place**
(`lumenairy/analysis/psf_mtf_otf.py:58` `_swap_halves_inplace`), `fft2` is
written out as its own two axis passes so the padded input can be released
between them, and ownership of the padded array is handed over explicitly
(`psf_mtf_otf.py:84` `_centred_fft2_take`) so the caller's name cannot keep it
alive across the transform.  `|amp|^2` is then built through one real buffer
and both normalisations are applied in place (`psf_mtf_otf.py:451`
`_scaled`).

Measured (interleaved medians, `tracemalloc` peak, `OPENBLAS_NUM_THREADS=1`,
2026-09-13), against the pre-fix expression restated in the same process:

| N_pupil | oversample | N_psf | padded grid | peak before | peak after | time before -> after |
|---|---|---|---|---|---|---|
| 512 | 2 | 1024 | 16.8 MB | 67.1 MB (**4.00x**) | 33.6 MB (**2.00x**) | 86.5 -> 65.6 ms (1.32x) |
| 512 | 4 | 2048 | 67.1 MB | 268.4 MB (4.00x) | 134.2 MB (2.00x) | 368.2 -> 347.3 ms (1.06x) |
| 1024 | 2 | 2048 | 67.1 MB | 268.4 MB (4.00x) | 134.2 MB (2.00x) | 362.4 -> 298.5 ms (1.21x) |

**Byte-identical**, by construction and by measurement.  A quadrant exchange
is a PERMUTATION: the FFT is handed a buffer holding bit-for-bit what
`ifftshift` would have put in a fresh copy, and the second exchange is
`fftshift` by definition.  Pinned over 12 shapes x 3 dtypes x 3 memory
layouts (`array_equal` on the raw bytes), including the odd lengths that fall
back to the explicit shifts.

**Why not the chessboard identity** the A7 design named.  `chess * fft2(chess
* a)` with `chess = (-1)^(i+j)` equals the shifted form up to a global sign
for any even length -- *mathematically*.  In floating point the two are
different reductions.  Measured over N = 2...1024 on three input families,
it is bit-identical **only when both lengths are a power of two**; on an even
non-power-of-two (N = 100 / 192 / 384) it agrees to ~5e-16 relative, which is
a moved default, and on an odd length it is a different array entirely
(relative 1.8 -- the cyclic shift the design warned about).  A quadrant
exchange moves no bits at any length.  The measurement is kept as a test so
the choice is re-derivable rather than asserted.

`compute_otf` shares the same helper and divides by its DC term in place.
Its own peak is unchanged at 2.50 grids -- NumPy makes its own complex copy
of a real input for the first FFT pass, and that floor is below what either
form reaches -- so what the rewrite removes there is the `fftshift` copy and
the `/ dc` copy, not the peak.

Files: `lumenairy/analysis/psf_mtf_otf.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py` (108 on this item).

### Added -- analysis/psf: `compute_psf(method='mft', dx_psf=...)`, the Soummer matrix Fourier transform (audit §15.9)

`compute_psf` gains a keyword-only `method='fft' | 'mft'` and, for `'mft'`, a
`dx_psf=` focal-plane pitch (`lumenairy/analysis/psf_mtf_otf.py:161`,
`:439`).  The `'mft'` path routes the Fraunhofer integral through
`lumenairy.propagators.fraunhofer_propagate_mft` (Soummer *et al.*,
*Opt. Express* **15** (2007) 15935), which samples directly onto whatever
pitch you name with **no padding at all**, so its cost scales with
`N_pupil + N_psf` rather than with the zoom.  The default stays `'fft'` and
is untouched.

Measured (512 pupil, 64 x 64 output window, warm, 2026-09-13):

| zoom | N_psf the FFT needs | fft peak | fft ms | mft peak | mft ms | ratio |
|---|---|---|---|---|---|---|
| 1 | 512 | 8.4 MB | 20.9 | 24.5 MB | 20.2 | 0.3x / 1.0x |
| 2 | 1024 | 33.6 MB | 70.6 | 24.5 MB | 20.4 | 1.4x / 3.5x |
| 4 | 2048 | 134.2 MB | 332.7 | 24.5 MB | 23.0 | 5.5x / 14.4x |
| 8 | 4096 | 536.9 MB | 1462.7 | 24.5 MB | 23.2 | 21.9x / 63.2x |
| 16 | 8192 | 2147.5 MB | 6520.6 | 24.5 MB | 18.6 | **87.7x / 349.7x** |
| 32 | 16384 | 8589.9 MB | 27797.0 | 24.5 MB | 22.7 | **350.9x / 1223.6x** |

The MFT column is flat because it does not care about the pitch.  On the FULL
natural grid the ranking reverses -- the Bluestein pads to `N_pupil + N_psf`
and transforms that, costing 3.1-7.5x the memory and 2.5-5.0x the time of the
plain padded FFT at oversample 4 / 2 / 1 -- which is why `'fft'` remains the
default and why the docstring says when to reach for the other one.

**Compatibility (the PSF grid contract does not move).**  With `dx_psf=None`
the MFT samples exactly the lattice the padded FFT delivers, and on EVEN
`N_pupil` and `N_psf` the two agree to **1.1e-15 of the peak** (measured over
N_pupil 64/128/256 x oversample 1/2/4 on an aberrated circular pupil, all
three `normalize` modes).  On an ODD length they differ by HALF A PIXEL and
nothing else -- `ifftshift` centres an odd axis on index `N // 2`, the MFT
follows the package convention `(arange(N) - N/2)*dx` -- and `method='mft'`
warns rather than handing back the shifted grid silently.

`normalize='power'` on the MFT path is the **analytic** Parseval constant
`(dx_pupil^2 / (lambda f))^2`, which is what the FFT path's empirical
in-window ratio evaluates to on the full grid.  That matters on a zoomed
window: an in-window rescale would force the visible fraction of the energy
to equal the whole pupil's.  `normalize='none'` divides the constant back out
to recover the FFT path's raw `|FT{pupil}|^2` convention.

Oracles, none of them produced by this library: a brute-force centred Fourier
sum (agreement **7.7e-15** at 1x, 4x and 11x zoom), the closed-form Gaussian
PSF on a grid where aliasing and truncation are both below 1e-200
(**1.1e-15 ... 2.2e-15**), and the analytic Airy pattern (8.3e-05, pixelation
limited -- the same at 1x and 8x zoom, which is the point: the residual is the
sampled aperture's, not the sampler's).

Files: `lumenairy/analysis/psf_mtf_otf.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Added -- analysis/psf: `encircled_energy_profile`, shared by the curve and the radius (A6)

`encircled_energy_curve` and `encircled_energy_radius` each build the same
sorted cumulative-energy profile, so a caller wanting BOTH -- the ordinary
spec-sheet pattern -- pays for two identical full-grid `argsort` passes.  The
construction is now a public value (`lumenairy/analysis/psf_mtf_otf.py:770`
`encircled_energy_profile(E, dx, *, dy=None, centroid=None) ->
(r_sorted, p_cum, r_max)`) and both functions accept it as `profile=`.

| N | two calls, unshared | one profile, shared | |
|---|---|---|---|
| 1024 | 221.7 ms | 105.0 ms | **2.11x** |
| 2048 | 1016.0 ms | 474.4 ms | **2.14x** |

at no change in peak memory, and with the curve and the radius coming back
**bit-identical** to the unshared path.  Passing the same profile to both is
also what makes the radius the exact inverse of the curve *by construction*
rather than by the two calls happening to agree; the round trip closes to
within one pixel-radius shell, which the regression computes from the profile
itself.

**No cache** (audit **§15.5**).  A content-keyed cache would have to hash
~67 MB per call at N = 2048 (~60 ms), a 12 % tax on the far more common
single-call path, and a key on anything less than the content is exactly the
incomplete-cache-key shape §15.5 describes.  The profile is a plain value: it
holds no reference to the field, is never cached, and is refused alongside a
`dy=` / `centroid=` that contradicts the ones frozen into it.

Files: `lumenairy/analysis/psf_mtf_otf.py`, `lumenairy/analysis/__init__.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.  The new entry point's
`_check_2d_scalar_field` guard is the package's 70th inventoried call site, declared
in `tests/unit/test_niche_audit_w4_input_kind.py` (wired at the release close, after
that file's fail-closed inventory caught it unwired).

### Performance -- analysis/zernike: the basis build shares one `rho ** k` memo, 1.61-2.27x, bit-identical (A6)

`zernike_basis_matrix` evaluated `_zernike_radial` mode by mode, and each mode
re-ran `np.power(rho, k)` -- a libm `pow` per pixel -- for every term.  The
first 21 modes ask for 34 such calls over 6 distinct exponents.  The build now
shares one `{k: rho ** k}` memo and one hoisted `rho <= 1` mask across the
modes, and accumulates each radial sum through one reused buffer
(`lumenairy/analysis/zernike.py:105`, `:240`, `:435`).

| N | modes | before | after | |
|---|---|---|---|---|
| 512 | 21 | 219.9 ms | 132.4 ms | 1.66x |
| 512 | 66 | 1058.9 ms | 507.2 ms | 2.09x |
| 1024 | 21 | 1043.9 ms | 599.7 ms | 1.74x |
| 1024 | 66 | 4426.4 ms | 1952.3 ms | **2.27x** |
| 2048 | 66 | 17521.4 ms | 8511.7 ms | 2.06x |

**Bit-identical** at every point (`array_equal` on the raw bytes of the basis
matrix and the pupil mask, against the per-mode loop restated in the same
process).  `N * R * angular` associates left, and that order is preserved
exactly -- reordering it to `(R * angular) * N` was measurably NOT
bit-identical and is the trap this item could easily have fallen into.

It is a time-for-memory trade, and the memory side needs no budget knob: the
memo is `n_max + 1` columns against the basis's `n_modes`, i.e.
`2 / (n_max + 2)` of the array the function already returns -- 0.33 at 15
modes, 0.29 at 21, 0.22 at 36, 0.17 at 66.  Measured peak 171.3 -> 221.1 MB
at N = 1024 / 21 modes (+29 %, the predicted 6/21) and 467.7 -> 550.5 MB at
66 modes (+18 %, the predicted 11/66).

Files: `lumenairy/analysis/zernike.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Fixed -- analysis/zernike: the radial factorial sum has lost nine digits by n = 22, where a recurrence now takes over (A6)

`_zernike_radial`'s closed-form sum alternates factorials of `n`, and the
cancellation costs about one decimal digit every two radial orders.  Measured
against an **exact rational oracle** -- the radial polynomial has integer
coefficients, so at `rho = k/128` its value is exactly rational and
`fractions.Fraction` evaluates it with no floating point and no library in it
at all -- the worst relative error over every `(n, m)`:

| n | 0-6 | 8 | 14 | 20 | 21 | **22** | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| factorial sum | **0.0** | 7.1e-15 | 1.8e-12 | 5.0e-10 | 8.9e-10 | **1.5e-09** | 7.3e-09 | 7.0e-06 | 3.2e-03 |
| Kintner recurrence | 0.0 | 1.1e-16 | 1.7e-15 | 1.9e-15 | 2.1e-15 | 2.0e-15 | 3.0e-15 | 2.8e-15 | 2.4e-15 |

`_zernike_radial` now hands orders `n >= 22` to the Kintner (*Opt. Acta* **23**
(1976) 679) recurrence in `n` at fixed `m`
(`lumenairy/analysis/zernike.py:123`).  22 is where the sum first passes 1e-9,
i.e. where it stops answering the question; `_ZERNIKE_RECURRENCE_MIN_N`
carries that table.  **Below it nothing moves** -- every mode any shipped
table (this module names modes to n = 8), any docstring, or any realistic
decomposition touches is `j < 253`, and the sum is still used there bit for
bit.  Stability limit, stated: the recurrence holds <= 3.9e-15 at every
`(n, m)` with `n <= 32` and <= 3.0e-15 out to `n = 40`.  It is also 1.04-1.56x
faster there, which is a bonus rather than the reason.

Files: `lumenairy/analysis/zernike.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Changed -- analysis/ao: the banded influence-function build is one helper, and a half-gigabyte eager DM cache now says so (A6)

Three things, none of which moves a number:

* the banded normal-equation accumulation is hoisted out of
  `DeformableMirror.fit_phase` into `DeformableMirror._banded_IF_apply`
  (`lumenairy/analysis/ao.py:249`), so there is ONE banded
  influence-function construction in the class rather than a copy per
  consumer;
* the three dense `np.meshgrid` sites become broadcast views
  (`ao.py:231` `_grid_views`, the S3-7 convention every sibling already
  carries) -- bit-identical, and 64 MB of avoidable transient per call site
  at N = 2048;
* an eager influence-function stack above `_IF_CACHE_WARN_BYTES` (half the
  `'auto'` ceiling) warns ONCE at construction, naming the bytes and the
  `cache_basis=False` escape.  The audit's own case -- a 16x16 DM on a
  512x512 grid -- is **exactly 536 870 912 bytes, i.e. exactly the
  inclusive `'auto'` ceiling**, so it caches, and it used to do so in
  silence.

The `'auto'` decision boundary itself is deliberately NOT moved: the cached
`phase()` is one `einsum` over the stack and the lazy one accumulates
actuator by actuator, which are different summation orders, so moving the
boundary would move the delivered phase map.  Both paths are pinned against a
local restatement.

Files: `lumenairy/analysis/ao.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Added -- sources: Gori pseudo-modes for Schell sources behind `generator='modes'` (Z3)

`_schell_phase_realizations` gains `generator='fft' | 'modes'` and
`n_pseudo_modes=`, forwarded by `create_gaussian_schell_source` and
`create_schell_model_source` (`lumenairy/sources/core.py:2153`, `:2183`,
`:2456`, `:2646`).  The default stays `'fft'` and is byte-identical.

A stationary field with Gaussian correlation `exp(-|d|^2 / (2 sigma^2))` is
exactly `phi(r) = M^(-1/2) sum_j exp(i(k_j . r + psi_j))` with
`k_j ~ N(0, sigma^-2)` per component and `psi_j` uniform, because the
characteristic function of that normal IS the target kernel.  There is no
grid in that statement and no transform, so the wrap the padded FFT
generator spends a 4-sigma pad suppressing (**Z2**) cannot arise at all, and
`E[<|phi|^2>] = 1` holds exactly rather than through a Parseval constant.
Each realisation is one `zgemm`: `phi = A @ B` with
`A = exp(i(y (x) k_y + psi))` `(Ny, M)` and `B = exp(i k_x (x) x)` `(M, Nx)`.

**The WP-A11 design's cost estimate does not survive measurement.**  It
priced the pseudo-mode sum at ~100x the padded FFT at M = 256, N = 512.
Measured here, per realisation at `sigma_g = L/8` with the heuristic M:

| N | M | fft ms | modes ms | ratio | fft peak | modes peak |
|---|---|---|---|---|---|---|
| 64 | 128 | 2.16 | 0.81 | **0.38x** | 1.98 MB | 0.86 MB |
| 128 | 128 | 8.85 | 2.03 | 0.23x | 7.88 MB | 2.11 MB |
| 256 | 128 | 37.89 | 4.26 | 0.11x | 31.47 MB | 6.30 MB |
| 512 | 128 | 154.14 | 11.43 | **0.07x** | 125.85 MB | 20.99 MB |

i.e. 2.6x to **13.5x cheaper** in time and 2.3-6.0x in peak memory.  The
estimate priced the FFT at `N` while the anti-wrap pad actually runs it at
2-5x `N` per axis.  The crossover on a 512 grid is near `M = 2300`
(18.7 ms at M = 256, 37.9 at 512, 69.3 at 1024 against the FFT's 154).

`M` defaults to the coherence-cell census `(Lx/sigma_g) * (Ly/sigma_g)`
(`sources/core.py:2129`), clamped into `[128, 4096]`.  Both constants are
derived, not chosen: for a random-phasor sum the intensity obeys
`E[I^2]/E[I]^2 = 2 - 1/M` **exactly**, against 2 for the circular-Gaussian
field the Schell model assumes, so M modes leave a contrast error of exactly
`1/M` and 128 puts it under 1 %.  The upper clamp is the cost cap and warns
with what it costs -- the k-space resolution of a single realisation, never
the ensemble kernel, which is exact at any `M >= 1`.

Verified: the realised correlation lands inside the sampling error of the
Gaussian target for both generators at `sigma_g = L/3` and `L/8`; a
chi-square of `|phi|^2` against `Exp(1)` on 20 equiprobable bins passes at
`M >= 128` (and the 'fft' generator is run through the same estimator as a
control); the exact `2 - 1/M` moment is matched at M = 8, 32, 128 and 512;
and the edge-to-edge correlation reads < 0.05 where the pre-Z2 periodised
path, reproduced in the same process, reads > 0.5 against a true 3.4e-14.

Files: `lumenairy/sources/core.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Performance -- sources: `create_gaussian_beam(geometry_dtype=np.float32)` for a complex64 beam, 2.00x -> 1.50x the output (Z3)

The optional half of the Z3 `create_gaussian_beam` item, which WP-A11
deferred because it changes the returned values.  It is now an opt-in with a
measured tolerance (`lumenairy/sources/core.py:384`); the default
(`geometry_dtype=None`, float64) is byte-identical over N in {17, 64, 256} x
three `normalize` modes x both complex dtypes x on- and off-axis centres.

At N = 2048 / complex64 (interleaved medians): peak **67.1 -> 50.4 MB**
(2.00x -> 1.50x the 33.6 MB output) and **93.6 -> 55.7 ms**.  Worst deviation
from the float64 geometry over N in {64, 512, 2048} x three `normalize` modes
x on- and off-axis: **1.19e-07 of the peak**, i.e. one float32 ULP of the
exponent -- which is all the `complex64` container can hold anyway.

It is refused with a `complex128` output: a double-precision request filled
from a single-precision exponent is a silent precision trap, and that is the
class of defect this audit exists for.  The returned `x` / `y` axes stay
float64 either way -- they are the caller's coordinates, not an intermediate.

Files: `lumenairy/sources/core.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

### Performance -- polarization: `apply_jones_matrix` 4.00 -> 3.00 full-grid complex arrays, bit-identical (Z3)

`J00*Ex + J01*Ey` written out builds two products and a sum per component, so
when the second component is formed the first component's result plus three
temporaries are live.  The mix moves into `_jones_mix_2x2`
(`lumenairy/elements/polarization.py:722`), where one scratch buffer serves
both components and both sums land in place.

Measured at N = 2048 complex128: **268.4 MB (4.00 grids) / 131.8 ms -> 201.3
MB (3.00 grids) / 96.4 ms**.

3.00 is the FLOOR, not a step towards the 2.00 the audit hoped for: the two
results must both exist at the end, none of the four products can be written
into a result before the other term of that result exists, and `field.Ex` /
`field.Ey` belong to the caller until the last product is read.

**Bit-identical**, and the gate is why.  `a += b` computes in
`result_type(a, b)` and then NARROWS to `a`, which is a different answer for
a mixed-precision `JonesField`; each in-place step is therefore taken only
when the dtypes already agree, and a mixed-precision field keeps the original
expressions.  The complex-multiply operand order is preserved everywhere
(NumPy's vectorised complex multiply is not bitwise commutative on this
build -- measured 1.8e-15 in the Z3 `stokes_parameters` work).  Pinned over
3 sizes x 2 dtypes x {plain, dark, NaN/inf, 1e-160 underflow} pixels, the
spatially-varying callable form, and the mixed-precision fall-back.

Files: `lumenairy/elements/polarization.py`.
Tests: `tests/unit/test_audit2609_b8_analysis_sources.py`.

---

**Migration notes:** none.  No default moves, in any of the seven items.
`generator=`, `n_pseudo_modes=`, `geometry_dtype=`, `method=`, `dx_psf=` and
`profile=` are all opt-in keywords whose defaults reproduce the previous
behaviour byte for byte, and the one value change in the library --
`_zernike_radial` at radial order `n >= 22` -- replaces an answer that the
exact rational oracle shows had already lost more than nine digits.

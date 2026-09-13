# VERIFY-B8 changelog text (verifier's additions to WP-B8)

### Fixed -- a Fortran-ordered PSF keeps its memory layout through `compute_otf` / `compute_mtf` / `compute_psf` (VERIFY-B8, audit A6.1)

WP-B8 replaced `fftshift(fft2(ifftshift(a)))` with an in-place quadrant exchange, which is bit-identical
by construction and measured so on 117 shape x dtype combinations here. What moved with it was the
*layout* of the result. `ifftshift` goes through `np.roll`, whose `empty_like` carries the input's order,
and NumPy's FFT carries that through to its output; `_centred_fft2`'s non-consuming copy was a plain
`a.copy()`, which is C-ordered whatever it was handed. So `compute_otf` on a Fortran-ordered PSF returned
an F-contiguous array before the change and a C-contiguous one after -- the same values in a different
buffer, which a byte comparison that normalises through `ascontiguousarray` cannot see. The copy is now
`a.copy(order='K')`, which is exactly what `np.roll` does, and the two are identical again in flags as
well as in bits. `lumenairy/analysis/psf_mtf_otf.py:158`. Peak transient is unchanged (2.00 full padded
grids for `compute_psf`, 2.50 for `compute_otf`, re-measured either side of the fix). Tests:
`tests/unit/test_audit2609_b8_analysis_sources.py::test_verifyb8_centred_fft2_keeps_the_input_memory_order`,
`::test_verifyb8_compute_otf_keeps_a_fortran_psf_fortran`.

### Fixed -- `encircled_energy_curve` / `encircled_energy_radius` validate a supplied `profile=` the way their docstrings said they did (VERIFY-B8, audit A6.2)

`encircled_energy_profile`'s Notes promised that the consumers "check what they cheaply can (shape and
monotonicity of the endpoints)" and `_resolve_ee_profile`'s own docstring listed "the two shapes, the two
lengths and the endpoints". Only the shapes and the lengths were checked. A `profile=` whose `p_cum` ran
1 -> 0 came back as `ee = [1, 0.75, 0.5]`; one running 0 -> 5 came back as `[0, 1, 1]`; a descending
`r_sorted` came back as `[0, 0, 1]`; an all-NaN `p_cum` was accepted. Nor was the profile's length
compared with `E.size`, so a profile built on an 8x8 field was accepted by a 16x16 call and quietly
answered the 8x8 question. Both checks are O(1) -- they cost nothing the argument was bought to avoid --
and both now run: `lumenairy/analysis/psf_mtf_otf.py:905` (length against `E.size`, after the
zero-power short circuit) and `:912` (the two endpoints). The endpoint slack is derived rather than
chosen: `p_cum` is a `cumsum` over `len(p_cum)` positive terms divided by their exact total, so
`len(p_cum) * eps` bounds its drift off 1 -- measured worst 8.8e-12 over N = 16...2048 x {Gaussian,
noise, Airy, near-delta} against `n eps` = 9.3e-10 at N = 2048, two decades of headroom, while every
shape the guard rejects is off by O(1). A profile built from a *different field of the same size* still
cannot be told apart in O(1) and is still accepted; the docstring now says that outright instead of
implying the opposite. Tests:
`::test_verifyb8_a_structurally_invalid_profile_is_refused`,
`::test_verifyb8_a_profile_from_a_differently_sized_field_is_refused`,
`::test_verifyb8_profile_endpoint_slack_admits_every_real_profile`.

### Fixed -- `create_gaussian_beam(geometry_dtype=np.float32)` is no longer defeated by a NumPy-scalar centre (VERIFY-B8, audit Z3 section 6.2)

Under NEP 50 a NumPy scalar is *strong*, so `X.astype(np.float32) - np.float64(x0)` promotes straight
back to float64. Passing `x0=np.float64(3e-6)` -- which is what a caller who read `x0` off another
array's coordinate axis will do -- therefore built the exponent in double precision after all: the
feature bought nothing, and it returned a *different field* from the same call spelled with a Python
float. Measured at N = 1024 with `dtype=np.complex64`: peak 16.80 MB (2.00x the complex64 output) with a
NumPy scalar against 12.61 MB (1.50x) with a Python float. The centre is now coerced with `float()`
inside the float32 branch only, so a Python float stays weak and the array's dtype survives;
`lumenairy/sources/core.py:561`. The default float64 geometry is untouched and byte-identical (3540
results re-run against `ed40e169^` and `ed40e169`). The documented tolerance is also corrected: it is
1.192e-07 of the peak on axis, exactly one float32 ULP as WP-B8 said, but up to **3.2e-07 off axis**,
because `(X - x0)` cancels in single precision -- measured over N in {64, 512, 2048} x three `normalize`
modes x centres out to 0.6 of the grid half-width. `lumenairy/sources/core.py:445-452`. Test:
`::test_verifyb8_float32_geometry_survives_a_numpy_scalar_centre`.

### Fixed -- `generator='modes'` refuses a degenerate `coherence_length` instead of answering (VERIFY-B8, audit Z3 section 6.1)

The Gori pseudo-mode generator draws its wavevectors with standard deviation `1 / coherence_length`, so a
zero, negative or non-finite coherence length has no meaning there -- but it did not say so. `0.0` raised
a bare `ZeroDivisionError` from inside `_gori_mode_count`, whose own `if not np.isfinite(cells) or
cells <= 0` guard can never run for that input because a Python-float divide by zero raises first; `nan`
returned an all-NaN ensemble with no warning; `inf` returned a single-phasor field; a negative value
silently returned the `|sigma_g|` field. The `'fft'` generator rejects three of those four with its own
errors, so the two disagreed silently. `_gori_mode_count` now tests `sigma_g` before dividing
(`lumenairy/sources/core.py:2140`) and the `'modes'` branch raises a `ValueError` with the
`CONVENTIONS.md` section 2 prefix naming the condition and why the draw is undefined (`:2341`).
**`generator='fft'` is the default and is not touched** -- it still accepts `coherence_length=0.0`, and
whether it should is a separate decision. Test:
`::test_verifyb8_modes_refuses_a_degenerate_coherence_length`.

### Changed -- the Kintner recurrence's stated stability envelope is re-measured on three grids (VERIFY-B8, audit A6.3)

`_ZERNIKE_RECURRENCE_MIN_N`'s block said the recurrence's own error is "<= 3.9e-15 at every (n, m) with
n <= 32 and <= 3.0e-15 out to n = 40". The second half cannot be true of the first -- a bound over n <= 40
contains the n <= 32 worst case -- and both were quoted from a single `rho = k/128` grid. Re-measured
against an exact-rational oracle (and cross-checked against a second, independent one: the Jacobi form
`R = (-1)^k rho^m P_k^(m,0)(1 - 2 rho^2)` evaluated in `fractions.Fraction`, 0 disagreements), absolutely,
since `|R_n^m| <= 1` on the unit disc: **3.907e-15** on `k/128` for n <= 32 *and* n <= 40, **1.51e-14** on
257 random `rho` in [0, 1], and **2.70e-14** on `rho = 1 - 10^-j` (j = 1..15), the hardest grid tried. The
block now quotes all three and calls it **3e-14 out to n = 40**. It also records that the same sweep runs
out to **n = 64 at <= 5.8e-14**, which closes WP-B8's deferred item 4 and is why the constant carries no
upper limit. `lumenairy/analysis/zernike.py:92`. Nothing the module computes changes -- the edit is
comment-only, and `record_history_fingerprints.py --check` reports
`lumenairy.analysis.zernike.md` unchanged, which is what that gate is for. The test bar (1e-13) already
had four decades of room and is unmoved.

### Changed -- `compute_psf`'s `method='mft'` compatibility statement is qualified where it does not hold (VERIFY-B8, audit section 15.9)

(Superseded in the same release by the orchestrator's ruling below: the FFT path now REFUSES `N_psf < N_pupil`, so the
divergence this entry describes is no longer reachable.  The paragraph records what VERIFY-B8 measured before the ruling.)

"With `dx_psf=None` the MFT samples exactly the lattice the padded FFT delivers" is true for
`N_psf >= N_pupil` and false below it, and the fault is on the FFT side: it pads only in the
`N_psf > N_pupil` branch, so for a smaller `N_psf` it silently returns an `N_pupil x N_pupil` array while
still reporting `wavelength*f/(N_psf*dx_pupil)` as its pitch (and, for `normalize='power'`, scaling by a
`psf_power_area` built from that pitch). `method='mft'` honours `N_psf`, so the two return different
SHAPES there -- measured at `N_pupil = 32`: `(32, 32)` against `(16, 16)`, with the same reported
`dx_psf`. The Notes now name the regime and which sampler is at fault, and the divergence is pinned so it
cannot be re-introduced as a surprise. `lumenairy/analysis/psf_mtf_otf.py:268`. **No behaviour moves** --
the FFT path's handling of `N_psf < N_pupil` is pre-existing and changing it would move a default; it is
recorded for the coordinator instead. Test:
`::test_verifyb8_mft_and_fft_shapes_diverge_below_the_pupil_size`.

### Changed -- the mixed-precision Jones fall-back is pinned on inputs that actually reach it (VERIFY-B8, audit Z3 section 6.3)

`_jones_mix_2x2` takes its in-place accumulation only while `Ex_new.dtype == scratch.dtype`, because
`a += b` computes in `result_type(a, b)` and then narrows to `a`. That guard is right, and it was
untested: deleting it left all 256 WP-B8 tests green. The existing
`test_b8_apply_jones_matrix_is_bit_identical_on_a_mixed_precision_field` cannot reach it, because
`np.asarray(matrix, dtype=complex)` makes every array-form Jones matrix `complex128` and NEP 50 then
promotes *both* products to `complex128` for any component dtype -- so nothing narrows and the fall-back
never engages. Its docstring described a fall-back its own inputs never trigger, which is the
`docs/TESTING_STANDARDS.md` S2 shape exactly; the docstring now says what the test really pins, with no
assertion removed. The case that does engage the gate needs a matrix narrower than a component -- a
`complex64` *spatially-varying* matrix against `Ex` complex64 and `Ey` complex128 -- and is now pinned,
with the narrowed accumulation evaluated in-process and asserted to differ, so the passing arm cannot be
satisfied by a build where the two happen to agree. No library change: the guard was already correct.
Test: `::test_verifyb8_apply_jones_matrix_falls_back_when_products_disagree`.

### Fixed -- `compute_psf(method='fft')` refuses `N_psf < N_pupil` instead of returning the pupil-sized array with the wrong pitch (orchestrator ruling on VERIFY-B8 section 4, audit A6.1)

The FFT sampler pads only when `N_psf > N_pupil`, so a smaller `N_psf` silently returned an `N_pupil x N_pupil` array while
`dx_psf` reported `wavelength*f/(N_psf*dx_pupil)` and, under `normalize='power'`, the scale carried a pixel area built from that
pitch -- off by `(N_pupil/N_psf)^2`.  Measured on the pre-fix library at `N_pupil = 32`, `N_psf = 16`: shape `(32, 32)` with the same
reported pitch as the MFT's honoured `(16, 16)`.  The call now raises `ValueError` with the CONVENTIONS section 2 prefix, naming the two
remedies (`N_psf >= N_pupil` / `oversample`, or `method='mft'`).  `lumenairy/analysis/psf_mtf_otf.py:361`.  **Migration:** a caller who
passed `N_psf` below the pupil size on the FFT path was receiving the un-cropped pupil-sized PSF with a mis-reported pitch; that call now
raises.  Ask for `N_psf >= N_pupil`, or use `method='mft'`, which samples exactly `N_psf` points.  Test:
`tests/unit/test_audit2609_b8_analysis_sources.py::test_verifyb8_the_fft_path_refuses_n_psf_below_the_pupil_size`.

# WP-A5 changelog text — propagator kernels (K1–K24)

Changelog-voice entries for the orchestrator to assemble. Audit:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §3, §3.1, §3.2.

---

### Fixed -- propagators/vector_diffraction: Richards--Wolf returned `E_z` with the wrong sign (K16, P0)

`richards_wolf_focus` returned the longitudinal component `E_z` with the
sign OPPOSITE to `E_x` / `E_y`, on every call, since the function was
written. `phi_p = arctan2(Yp, Xp)` is the APERTURE-point azimuth, while
the textbook aplanatic strength vector is written in the RAY-DIRECTION
azimuth `phi_ray = phi_p + pi`; `e_x` and `e_y` are even under
`phi -> phi + pi` and were unaffected, but `e_z` is odd, so the `-` in
`Pz = P*(-px*cp*s - py*sp*s)` (`vector_diffraction.py:347`) delivered the
negative of the correct value. `|E_z|^2` is blind to the sign, so
`debye_wolf_psf`, the GUI dock and every existing test were blind: no
test pinned it. Wrong before this release: focal-field handedness and
spin angular momentum, `S3` of the focal field, optical-force and
spin--orbit calculations, and any coherent superposition using all three
components — i.e. the entire reason the function returns `E_z`
separately.

Measured against an independently derived Debye--Wolf oracle (rigid
rotation of the pupil polarisation reduced to the radial Bessel form
`E_z/E_x = -2i I01/(I00+I02)`, `scipy.integrate.quad`; x-polarised
uniform pupil, NA 0.5, f = 4 mm, lambda = 633 nm, Np = 256), per-component
code/oracle ratios at `x_f` = 0.60 / 1.21 / 1.81 / 3.01 um:

| component | before | after |
|---|---|---|
| `E_x` | +1.0006 / +0.9954 / +0.9994 / +0.9670 | unchanged |
| `E_z` | **−1.0006 / −0.9954 / −0.9994 / −0.9670** | **+1.0006 / +0.9954 / +0.9994 / +0.9670** |

`Im(E_z/E_x)` just off axis at `x_f > 0` moves from **+0.6966** to
**−0.6966**, which is the sign Novotny & Hecht eq. 3.66 requires.
Symmetries are untouched: `E_z` odd in `x` to 4.5e-16, `E_x` even to
2.4e-16, `E_z` on the `y` axis 3.1e-19 of its own maximum.

Files: `lumenairy/propagators/vector_diffraction.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK16RichardsWolfEzSign`
(4 tests).

**Migration.** Any code that consumed `E_z` from `richards_wolf_focus`
coherently — a three-component superposition, a Stokes/spin calculation,
an optical-force integral — was wrong by the sign of that component and
now is not. Intensity-only consumers (`debye_wolf_psf`, `|E|^2` maps) are
bit-identical.

---

### Added -- propagators/vector_diffraction: the `pupil` coordinate convention is documented (K10, P3)

Two auditors reached opposite conclusions about `richards_wolf_focus`'s
pupil indexing because the docstring never stated it. It is the PHYSICAL
EXIT-PUPIL (aperture) coordinate, not the projected ray direction; the
two differ by a point inversion, invisible for a 180-degree-symmetric
pupil and immediately wrong for coma, tilt, a decentred sub-aperture, a
segmented aperture or a metasurface pupil. The convention is now stated
in the `pupil` parameter docstring with the measurement that pins it (a
pupil ramp `exp(+2j pi u x_p)` moves the focus to `x_f = +u lambda f`;
measured +2.3600 um against a predicted +2.3737 um), and the `Returns`
section states `E_z`'s symmetry and sign.

Files: `lumenairy/propagators/vector_diffraction.py`.
Tests: `...::TestK16RichardsWolfEzSign::test_k10_pupil_is_indexed_by_the_aperture_coordinate`.

---

### Fixed -- propagators/rs: the Rayleigh--Sommerfeld kernel created energy in the near field (K9, P0)

`rayleigh_sommerfeld_propagate` point-sampled the RS Green's function on
the padded `2N` grid. Its local phase gradient `k sin(theta) dx` exceeds
the `pi`/pixel Nyquist limit whenever `z < 2 N dx^2 / lambda`, and
nothing checked it — so on ordinary grids, with all-default arguments,
the convolution CREATED energy. Reached by `propagate(method='rs')`,
`propagate(method='hf')` and `propagate_huygens_fresnel` (a three-line
delegation to RS). `bandlimit=True` was all-pass in exactly that regime
(its cutoff exceeds the grid Nyquist under the same algebraic condition)
and made things five decades worse outside it.

`kernel` (new, `{'auto', 'transfer', 'spatial'}`, default `'auto'`) now
routes between two discretisations of the same RS-I operator:
`'transfer'` evaluates the exact RS-I transfer function
`exp(i k z sqrt(1 - (lambda f)^2))` analytically in the FREQUENCY domain
on the padded grid (with the evanescent set zeroed), and `'spatial'` is
the historical point-sampled build. `'auto'` selects `'transfer'` below
`2 N dx^2 / lambda` and `'spatial'` at and above it, so **every call at
or above the threshold is bit-identical to v5.45** and only the regime
the audit measured as broken is re-routed. `'spatial'` now RAISES inside
its alias regime rather than returning the wrong field.

Measured against an exact Hankel angular-spectrum oracle (Gaussian
w0 = 6 um, lambda = 633 nm, z = 50 um):

| grid | `P_out/P_in` before | after | relative L2 before | after |
|---|---|---|---|---|
| N = 64, dx = 2 um | 21.44 | **1.000000** | 4.50 | **5.3e-8** |
| N = 128, dx = 1 um | 5.31 | **1.000000** | 2.08 | **6.1e-8** |
| N = 128, dx = 2 um | 25.70 | **1.000000** | 4.95 | **5.3e-8** |

Far field unchanged and bit-identical: 6.1e-8 / 6.3e-8 / 6.8e-8 /
4.4e-8 at z = 200 um / 300 um / 1 mm / 3 mm on the N = 128 / dx = 1 um
probe. The two branches converge onto each other as the grid is refined
— at `z = z_crit` the step between them is 6.3e-5 / 1.0e-9 / 3.8e-14 at
(N, dx) = (64, 0.50 um) / (128, 0.40 um) / (256, 0.25 um), always well
below each arm's own error at that grid.

Files: `lumenairy/propagators/rs.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK9RayleighSommerfeldNearField`
(17 tests).

**Migration.** No signature change for existing calls. Near-field RS
results (`z < 2 N dx^2 / lambda`) change — they were wrong. Pass
`kernel='spatial'` to force the historical build; it raises rather than
aliasing.

---

### Changed -- propagators/rs: the docstring now says what was measured (K8, K15, P3)

Three claims were measured false and are replaced with the measurements:

* "Near-field propagation (z ~ a few wavelengths) where ASM's
  band-limiting can suppress valid high-frequency content" — the
  opposite: at z = 50 um on the N = 128 / dx = 1 um probe, ASM with its
  default `bandlimit=True` measures 6.1e-8 while the pre-v5.46 RS spatial
  kernel was 2.08 (208 %) wrong.
* "For intermediate distances they agree to machine precision when ASM
  uses no band-limiting" — the RS-vs-ASM difference plateaued at 2.8e-2
  (the point-sampling floor), not zero.
* The `bandlimit` parameter is documented as NOT a near-field remedy,
  with the measurement (4.4e-8 vs 1.8e-2 at z = 3 mm).

RS's real advantage over single-grid ASM is now stated correctly: the
zero padding makes it a LINEAR convolution, so it does not wrap energy
around the grid (measured 4.4e-8 vs 7.6e-1 at z = 3 mm).

Files: `lumenairy/propagators/rs.py`.

---

### Added -- propagators/fresnel, propagators/mft: a chirp-sampling guard on the single-FFT Fresnel kernels (K1, P1)

`fresnel_propagate` samples the quadratic chirp `exp(i k r1^2/(2z))` at
the grid pitch; the sum is a valid quadrature only for
`z >= max(Nx dx^2, Ny dy^2)/lambda`, and nothing checked it. Measured
against an 8x-oversampled direct Fresnel quadrature (N = 128, dx = 2 um,
lambda = 633 nm, smooth Gaussian so the FIELD is fully sampled): relative
error 3.34e-1 / 3.43e-1 / 3.92e-2 at three output points at
`z = 0.25 z_crit`, with **zero** warnings. `fresnel_propagate_mft`
evaluates the same aliased sum (the two agree to 3e-14 at every z, which
is why the MFT sibling is not an oracle for this) and gets the same
guard.

A `RuntimeWarning` naming `angular_spectrum_propagate` now fires below
the bound; values are unchanged. Measured after: 2 warnings at
`z = 0.25 / 0.50 z_crit`, 0 at `z >= z_crit`.

Files: `lumenairy/propagators/fresnel.py`, `lumenairy/propagators/mft.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK1FresnelChirpGuard`
(5 tests, including the counter-pin that it stays silent where the kernel
is valid).

---

### Fixed -- backend/scipy: `jv(v, x)` computed `scipy.special.jv(x, v)` (K2, P1)

`lumenairy.backend.scipy.jv` passed the array as the FIRST positional
argument of a two-argument special function, transposing order and
argument, while the JAX branch three lines above was correct — so the two
backends disagreed and the NumPy branch returned a plausible wrong
number. `_dispatch_special` gains an `arg_pos` keyword so the backend
selector no longer has to be the first argument.

| call | before | scipy / after |
|---|---|---|
| `jv(0, 2.0)` | +0.000000000 | +0.223890779 |
| `jv(1, 3.0)` | +0.019563354 | +0.339058959 |
| `jv(2, 1.5)` | +0.491293779 | +0.232087672 |
| `jv(0.5, 4.0)` | +0.000160736 | −0.301920513 |
| `jv(1, [1,2,3])` | [0.44005, 0.11490, 0.01956] | [0.44005, 0.57672, 0.33906] |

No in-library consumer was affected (`elements/bor/*` import `jv` from
`scipy.special` directly), so this was latent — but it is a module-level
function in the public backend namespace.

Files: `lumenairy/backend/scipy.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK2BackendBesselArgumentOrder`
(5 tests).

---

### Fixed -- propagators/sas: single-precision SAS lost its ASM−Fresnel correction to cancellation (K3, P1)

`scalable_angular_spectrum_propagate` built its frequency axes and kernel
phase arguments in the caller's real dtype, so a complex64 input formed
`h_AS - h_Fr` — a cancellation of two quantities near 1 — in float32 and
then multiplied it by `k z`. Measured `max|Delta(h_AS - h_Fr)|` float32
vs float64 = **9.091e-08**, i.e. 2.9e-3 rad of phase error at
z = 3.24 mm and **0.902 rad at z = 1 m** — and long distance is SAS's
reason to exist. This contradicted the library's own complex64 contract
("does not degrade with phase magnitude").

The frequency axes and both chirp coordinate grids are now built in
float64 unconditionally (the f64-carrier-then-cast recipe `fresnel.py`
already uses), and the difference is evaluated through its
cancellation-free closed form `-u^2/(2(1+sqrt(1-u))^2)`. Only the
finished complex kernels are cast; the returned dtype is unchanged.

Measured end-to-end, complex64 field against its complex128 twin, max
phase error over the bright region:

| z | before (kernel-level) | after |
|---|---|---|
| 3.24 mm | 2.9e-3 rad | **1.50e-5 rad** |
| 1 m | **0.902 rad** | **6.76e-6 rad** |

i.e. the error no longer grows with `k z` — it is flat at the float32
FFT noise floor. In float64 the closed form is also exact where the
subtraction lost everything below `u ~ 1e-4`; the resulting change to
existing complex128 results is 1.7e-9 rad at z = 1 m.

Files: `lumenairy/propagators/sas.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK3SasSinglePrecisionCancellation`
(2 tests).

---

### Performance -- propagators/asm: cap the transfer-function build's workspace (K5, P2, byte-identical)

`_get_asm_H_natural` sized its row chunk at 10 % of the RAM budget, which
resolves to the WHOLE grid below N ~ 8192 on a large box, so the float64
kernel workspace was built full-grid. The streamed sibling already capped
its band in elements for exactly this reason, but only for itself. A new
`_ASM_H_BUILD_BAND_ELEMS = 1 << 18` caps the plain builder too, so the
default path, every cold `_H_CACHE` build and the batch variant get it.

Measured (tracemalloc, complex128, bandlimit on; one grid = 67.1 MB at
N = 2048): H-build transient **4.06 -> 1.26 full grids** at N = 2048 and
4.06 -> 2.03 at N = 1024, **byte-identical** output at both. Timing is
flat (211.0 ms vs 222.3 ms median of 3 at N = 2048). At N = 32768 the
kernel workspace goes from ~8.6 GB to ~134 MB.

Not reproduced: the audit's implied improvement to the FULL cold ASM
call. Measured in a cold process at N = 2048 the peak is 6.55 grids
either way — with warm plans the H build is no longer the peak, and in a
cold process the pyFFTW aligned-buffer allocation dominates.

Files: `lumenairy/propagators/asm.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK5AsmKernelWorkspaceCap`
(2 tests).

---

### Performance -- propagators/fft_infra: one lock per pyFFTW ping-pong slot (K4, P2)

`_build_plan_entry` allocated one `pyfftw.FFTW` plan per buffer but a
single `threading.Lock` for the entry, and `_fft2` / `_ifft2` /
`_fft2_nd` / `_ifft2_nd` held that one lock across copy-in and execute.
The slot index is already advanced under `_PYFFTW_PLAN_LOCK`, so two
threads always receive different plans and different buffers — there was
nothing for them to race on, yet they serialised. Measured max
simultaneous threads inside the pyFFTW critical section with 4 threads x
6 calls on one (1024, 1024) complex128 shape: **1 before, 2 after** (the
`n_bufs = 2` ceiling). The per-slot lock still guards the one real hazard
— two callers wrapping around to the same slot.

Files: `lumenairy/propagators/fft_infra.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK4PerSlotPlanLocks`.

---

### Fixed -- propagators/fft_infra: the pyFFTW failure blacklist is keyed on (shape, dtype, direction) (K7, P2)

`_PYFFTW_BAD_SHAPES` recorded the bare shape, so one complex128
`MemoryError` at (512, 512) also blacklisted complex64 at the same shape
— half the memory, likely to succeed — and the inverse direction, which
has its own plan and buffer. Measured: after one simulated failure the
blacklist was `{(512, 512)}` and a subsequent complex64 transform skipped
pyFFTW. Entries are now `(shape, dtype.str, direction)` triples, matching
the plan cache's own key minus `threads`; the warning names all three.

Files: `lumenairy/propagators/fft_infra.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK7PyfftwBlacklistKey`.

---

### Fixed -- propagators/asm, fft_infra, result: three small correctness / contract gaps (K8, P3)

* **`_build_asm_H_square`'s "bit-exact" contract was false for odd N.**
  It formed the frequency axis by DIVISION while the shared
  `_get_or_make_freq_grids` multiplies by the reciprocal — up to 1 ULP
  apart whenever `1/(N dx)` is inexact. Measured `max|dH| = 9.096e-13` at
  N = 255 / dx = 0.5 um / `bandlimit=False`; now **0** at every parity
  tested (N = 64, 127, 129, 255, 256). `_get_or_make_bandlimit`'s masks
  use the same expression, closing the latent mask/kernel label
  mismatch. This is the `shack_hartmann` per-lenslet path.
* **The `_H_CACHE` handed its entries out writeable.** Two successive
  `_get_asm_H_natural` calls at one key return arrays sharing memory, and
  "callers must not mutate it in place" was a comment. Entries are now
  stored read-only; the public `return_transfer_function=True` return
  still copies and stays writeable.
* **`PropagationResult.__array__` rejected numpy 2's `copy=` keyword.**
  On numpy 2.4.6 `np.array(result, copy=True)` emitted a
  `DeprecationWarning` and `np.array(result, copy=False)` raised
  `ValueError`. Both now work.
* The `_asm_H_from_kz` complex64 comment claimed an accuracy win the
  measurements do not support (the naive `astype(complex64)` IS the
  correctly-rounded value; the mod-2*pi fold is 1.58x WORSE at
  `k z = 6e8`). The code is kept — it is genuinely needed on the JAX-x32
  path and uses less transient memory — and the comment now says so.

Files: `lumenairy/propagators/asm.py`, `lumenairy/propagators/fft_infra.py`,
`lumenairy/propagators/result.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK8BitExactnessAndCacheHygiene`
(8 tests).

---

### Fixed -- propagators/hf, propagators/mhs: the resample renormalisation fabricated energy across a crop (K11, P1)

`propagate_huygens_fresnel_freespace` and
`mhs.prescription_subdomain(method='maslov')` each carried a verbatim
copy of a `sqrt(p_in/p_out)` Parseval renormalisation that conflated two
different things: the small interpolation drift of a bicubic
`map_coordinates` (which should be corrected) and a genuine physical CROP
when the requested window is smaller than the source's (which must not
be). Measured on
`propagate_huygens_fresnel_freespace(E, 1e-3, 633e-9, 2e-6, output_dx=0.5e-6)`
at N = 64: the requested +-16 um window genuinely contains **67.27 %** of
the native-grid power and the returned array carried **100.00 %** —
amplitudes inflated **1.219x**, intensities 1.486x.

One shared helper `_resample_preserving_window_power` now measures the
reference power on the SOURCE grid restricted to the area the output
pixels tile, so the correction is the interpolation drift alone, and a
crop discarding more than a part in 1e6 is reported as a
`RuntimeWarning` naming the retained fraction. When the target window
covers the whole source the restriction is the identity and behaviour is
unchanged. The duplicated block and its three stale cross-references are
gone.

Files: `lumenairy/propagators/hf.py`, `lumenairy/propagators/mhs.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK11ResampleDoesNotFabricateEnergy`
(2 tests, including the counter-pin for the uncropped case).

---

### Fixed -- propagators/hf: a 0.5 % pitch change was a silent no-op (K21, P2)

`propagate_huygens_fresnel_freespace`'s same-grid short-circuit compared
pitches with `np.isclose(dx, target_dx, rtol=1e-12)`, which still carries
numpy's default `atol = 1e-8` — **10 nm in this library's metres**. A
0.5 % pitch change at 1 um and a 10 % change at 100 nm both compared
equal, so the un-resampled field came back LABELLED with the requested
pitch: the "wrong sampling metadata" class the dispatcher raises for
elsewhere. Now `abs(dx - target_dx) <= 1e-12*dx`; a genuine no-op still
short-circuits bit-for-bit.

Files: `lumenairy/propagators/hf.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK21PitchGate`
(3 tests).

---

### Fixed -- propagators/dispatch: `propagate(method='hf')` returns an ndarray like every other method (K20, P2)

The `hf` free-space branch was the only output-grid-capable method that
changed its RETURN TYPE — a bare ndarray with no grid kwargs, an
`(E, dx)` TUPLE with either of them, including an `output_grid` equal to
the input (a strict no-op) — while `asm` / `gbd` / `hfpi` returned an
ndarray in both cases, and the dispatcher's own W9-4 message steers
callers to `method='hf'`. With `return_result=False` — what
`mhs.prescription_subdomain` uses — the tuple leaked. The dispatcher now
unpacks it; the requested pitch is the caller's own argument.

Files: `lumenairy/propagators/dispatch.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK20UniformReturnType`
(4 parametrised), and `tests/unit/test_v5_3_hf_freespace_output_grid.py`
(3 tests updated — they PINNED the inconsistency).

---

### Changed -- propagators/hf: `chunk_output` is un-deprecated and honestly sized (K22, P2)

`chunk_output` was deprecated in v5.17 as a no-op with a note naming the
fix and then removing it. It is restored as a genuine output-batch size:
the output coordinates are broadcast to `(n_chunk, 1, 1)` so one
vectorised `opl_fn` evaluation covers a whole block of output pixels. A
one-shot probe detects a callable that cannot take the batched form (the
pre-v5.46 contract said the output coordinates are scalars) and falls
back to the per-pixel path with a `RuntimeWarning`. Output is
**bit-identical** for every value of `chunk_output`, with and without Van
Vleck.

**The audit's projected gain is not reproducible and the docstring says
so.** Batching removes Python-level DISPATCH, which is a small share of a
memory-bandwidth-bound computation; measured ladder on this workstation
(medians of 5 interleaved runs, ms per output pixel, Van Vleck on,
complex128):

| N_in | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| 64 | 0.401 | 0.360 | **0.323** | 0.328 | 0.449 | 1.634 |
| 128 | **1.450** | 1.465 | 2.099 | 7.077 | 6.422 | 6.419 |
| 256 | **9.563** | 30.45 | 28.02 | 29.50 | 30.34 | 28.82 |

The best available gain is **1.24x at N_in = 64**, and a batch whose
working array leaves the L2 cache is 3–5x SLOWER. The auto rule therefore
targets a ~128 KB working array, which selects the measured optimum at
each of those three sizes. (The audit's "119 min -> minutes for 256^2" is
not achievable by batching; this workstation measures 9.56 ms/px at
N_in = 256, i.e. ~10 min for a full 256x256 output, and the docstring
points at the Fourier routes instead.)

Also on this path: the JAX branch accumulates batches into a Python list
and assembles once instead of allocating a full output array per output
pixel, and a complex64 caller now gets its kernel built in single
precision after a float64 mod-one-cycle fold of `Phi` (which is in waves
and of order `z/lambda`), instead of a complex128 grid built and thrown
away.

Files: `lumenairy/propagators/hf.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK22HfQuadratureChunking`
(4 tests); `tests/unit/test_audit_w6_propagators.py::TestP357ChunkOutputDeprecated`
updated (it pinned the deprecation).

---

### Fixed -- propagators/hfpi, vectorial_hfpi: `wavelength` is keyword-required (K12, P1)

`apply_aperture_diffraction` and `apply_vector_aperture_diffraction` took
`wavelength: float = 0.0`, and the `1/(i lambda)` Kirchhoff prefactor was
gated on `wavelength > 0` — so omitting it silently DROPPED the physics.
Measured: every path weight wrong by exactly `1/lambda` = 1.5798e6 in
magnitude AND by −90 degrees in phase, with zero warnings — precisely the
failure the v4.11.2 prefactor work fixed. Both functions are in `__all__`
and are the documented way to build a custom cascade by hand.
`wavelength` is now keyword-required and raises on non-positive or
non-finite values.

Files: `lumenairy/propagators/hfpi.py`,
`lumenairy/propagators/vectorial_hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK12WavelengthIsRequired`
(2 tests).

**Migration.** Add `wavelength=` to any direct call. Every in-library
call site already passed it; two validation call sites were updated.

---

### Fixed -- propagators/hfpi, vectorial_hfpi: the HFPI estimator is now the Huygens--Fresnel integral (K13, K18, P1)

Two compounding normalisation defects made the returned amplitude depend
on the OUTPUT PIXEL AREA and the SOURCE PIXEL COUNT, so merely rebinning
the grids changed the answer with no physics change, and `|E|max` moved
14x between 2 M and 8 M paths.

* **K18 — the source-area factor.** The source pixel is drawn uniformly
  over `Ny*Nx` pixels, so the unbiased estimate carries the whole
  illuminated area `Ny*Nx*dx^2`; the code applied ONE pixel's `dx^2`.
  Measured `sum(weights)/exact` against the closed-form source term:
  1.000004 (1x1), 0.062140 (4x4), 0.003945 (16x16), 0.000260 (64x64) —
  tracking `1/N_pix`, i.e. **4096x low at a 64x64 source**. After:
  0.999985 / 1.0106 / 0.9515 / 1.2089, all consistent with 1 to the
  Monte-Carlo noise of the probe.
* **K13 — the output-binning Jacobian.** The exact bias law
  `E[HFPI]/E_true = dx_out^2 cos(theta)/(N_src_px r)` was derived and
  confirmed to MC noise (meas/pred 1.0068 and 0.9950 at z = 2 / 4 mm with
  24 M paths at occupancy 1.000). Each landed path is now weighted by
  `r/(dx_out^2 cos(theta_out))` at bin time, which supplies the missing
  `1/r` spreading and the pixel-solid-angle conversion together.
  `PathBundle` / `VectorPathBundle` gain a `leg` field carrying the
  GEOMETRIC distance since the last emission (`opl` cannot serve: it is
  the OPTICAL path on free-space legs and the absolute accumulated `opd`
  on the prescription walk).

Measured end-to-end against band-limited ASM (exact there: 6.1e-8 against
the Hankel oracle) on a Gaussian w0 = 12 um, N = 32, dx = 4 um, z = 2 mm,
read out as the unbiased least-squares complex scale:

| | before | after |
|---|---|---|
| HFPI / ASM | 7.8e-12 (`dx^2/(N_pix z)`) | **0.976 – 1.012** across seeds and 0.5–8 M paths |

`accumulate_to_grid` / `accumulate_vector_to_grid` gain
`normalisation={'physical','legacy'}` (default `'physical'`), and refuse
`'physical'` with an actionable message on a bundle whose landed paths
have travelled zero distance instead of silently returning zeros.
`propagate_hfpi_through_prescription` defaults to `'legacy'` and now
WARNS, because that walk bins at the last surface rather than propagating
to a separate output plane, so the last leg has zero length and the
Jacobian is undefined there (deferred, see the report).

The 38-line normalisation warning on `propagate_hfpi` was itself wrong —
it listed "the source pixel area `dx^2`" under *does apply* — and is
rewritten to the corrected accounting.

Files: `lumenairy/propagators/hfpi.py`,
`lumenairy/propagators/vectorial_hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK18SourceAreaNormalisation`
(3 parametrised), `::TestK13BinningJacobian` (2 tests).

**Migration.** **Returned HFPI amplitudes change by orders of magnitude.**
Phase structure (fringe positions, interference contrast) is unaffected.
Pass `normalisation='legacy'` to restore the raw path sum. Code that
re-normalised HFPI against an ASM reference should drop that step.

---

### Fixed -- propagators/hfpi, vectorial_hfpi: `rng=None` draws fresh entropy (K19, P1)

`_spawn_rng`'s `None` branch is documented as "let each aperture pull
from system entropy", but all five consumers wrote
`RandomState(rng if rng is not None else 0)`, so the DEFAULT was the
fixed seed 0 and that branch was unreachable. Measured: two default runs
byte-identical, and identical to `rng=0`. HFPI is a `1/sqrt(N)`
Monte-Carlo estimator sold on that convergence, and the canonical way to
see its error is to re-run with a new seed — on the default path that
error estimate was identically ZERO, so any ensemble / tolerancing /
seed-averaging loop that did not vary `rng` reported a spuriously tight
spread. All five sites now pass `rng` through unchanged.

Files: `lumenairy/propagators/hfpi.py`,
`lumenairy/propagators/vectorial_hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK19RngDefaultDrawsEntropy`
(3 tests, including the counter-pin that an explicit seed is still
reproducible).

**Migration.** Pass an explicit `rng=` for reproducible runs. Every
existing determinism test in the suite already did.

---

### Fixed -- propagators/hfpi: `_spawn_rng` is a pure function of (parent, stream) (K24, P3)

The `np.random.Generator` branch called `rng.spawn(stream_index + 1)[-1]`,
which MUTATES the caller's generator on every call and discards
`stream_index` children to use one — so "stream 1 drawn after stream 0"
differed from "stream 1 drawn alone". It now derives the child from the
parent's SeedSequence entropy, exactly as the `int` branch does. The JAX
branch's `except Exception: pass` — which fell through to "return as-is",
handing BOTH streams the identical key, the exact correlation the
function exists to prevent — is narrowed and now warns.

Files: `lumenairy/propagators/hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK24SpawnRngIsAPureFunction`
(2 tests).

---

### Added -- propagators/vectorial_hfpi: actual vector physics (K17, P1)

The module's header advertised "the m-theory dipole obliquity tensor for
vector-correct secondary-source amplitudes" and listed high-NA imaging,
cascaded polarizing elements and birefringent elements as cases that
REQUIRE it. No such tensor existed: the default path multiplied both
Jones components by the same scalar, i.e. by the identity as far as
polarisation is concerned. Measured — 24x24 source, two 200 um legs
around a 40 um aperture, 60 000 paths, same seed — the vector `Ex` output
was bit-identical to a scalar HFPI run on `Ex_in`
(max|diff| = 1.9e-23), and a 45-degree linear input showed
`|Ey/Ex - 1| <= 1.1e-16` over the WHOLE output grid: zero depolarisation,
anywhere, at twice the cost of the scalar propagator. The opt-in
`vector_projection=True` did rotate, but discarded the longitudinal
component it created (8.8 % of the incident `|E|^2` at a 0.8 rad cone),
multiplied by the scalar obliquity a second time, and never projected at
emission.

Implemented instead: `VectorPathBundle` carries `Ez`, and at emission and
at every re-emission the three-component field is carried onto the path's
own transverse plane by the RIGID ROTATION taking the previous
propagation direction to the new one (Rodrigues; for `s_from = +z` it
reduces term by term to the Richards--Wolf aplanatic
`R_z(phi) R_y(theta) R_z(-phi)`). That rotation is orthogonal, so it adds
no amplitude and cannot double-count the obliquity; the scalar Kirchhoff
obliquity is applied exactly once. `Ez` is accumulated alongside `Ex` /
`Ey` and returned on request.

Measured after (24x24 source, 0.35 rad cone, 200 000 paths):

| | before | after |
|---|---|---|
| `|E|^2` preserved by the projection | 0.841 (15.9 % dropped) | **1 − 1.1e-15** |
| transversality `max|E'.s|` | n/a | **4.6e-16** |
| x-pol: `|Ez|^2` fraction | **0** | **1.40e-2** |
| x-pol: cross-polarised `|Ey|^2` fraction | **0** | **9.3e-5** |
| 45-degree: `max|Ey/Ex − 1|` over the grid | **1.3e-16** | **0.143** |

The module docstring is rewritten: the "m-theory dipole obliquity tensor"
claim is deleted, what the module does and does NOT do is stated, and
high-NA vector FOCUSING is routed to `richards_wolf_focus`.

Files: `lumenairy/propagators/vectorial_hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK17VectorialHfpi`
(5 tests, including the counter-pin that `vector_projection=False` still
reproduces two scalar runs to 1e-12).

**Migration.** `vector_projection` defaults to `True` and the operation
it names changed. `vector_projection=False` reproduces the pre-v5.46
behaviour exactly. Pass `return_ez=True` for the longitudinal component;
the historical 2-tuple return is unchanged by default.

---

### Fixed -- propagators/hfpi, vectorial_hfpi: guards, caps and reachable levers (K14, K23, P2)

* **`cone_half_angle` reaches the free-space entry points.** The v5.31
  under-sampling guard's own message recommends narrowing it "from its
  ~90-degree default toward the angle the output grid actually
  subtends", and passing it raised `TypeError` on
  `propagate_hfpi_freespace_aperture`,
  `propagate_vector_hfpi_freespace_aperture`, `propagate_hfpi` and
  `propagate(method='hfpi')` — only the prescription walk exposed it. It
  is now accepted and threaded on both free-space entry points.
* **The vector accumulator shares the guard.** v4.13.1 forked it for
  index sharing, so the v5.31 guard never ran there: measured on
  identical geometry, 20 000 paths onto a 64x64 grid gave 2 non-zero
  pixels (0.05 %) with the scalar path warning and the vector path
  silent, and the vector entry point accepted no `on_undersampled`
  kwarg. Both accumulators now call one `_check_landed` helper and share
  `_bin_paths`; the vector entry point accepts `on_undersampled`.
* **`n_paths` is a cap in `init_paths_stratified`.** `n_per = max(1,
  n_paths // n_total)` clamped to 1 and `n_paths_actual = n_per*n_total
  >= n_total` regardless of `n_paths`, so an explicit stratification
  allocated up to **10 486x** the requested paths (measured 1 048 576
  from an `n_paths=100` call at (32,32)/(32,32), 76.6 MB; 16.8 M and
  ~1.2 GB at (64,64)/(64,64)), and the `[:n_paths_actual]` truncation was
  always a no-op. The docstring's sub-sampling ("sample only `n_paths`
  strata uniformly without replacement") is now implemented.
* **CuPy.** `accumulate_to_grid`'s guard and `_hfpi_segment_trace` route
  through `to_numpy` unconditionally instead of `np.asarray`, which
  raises on a CuPy device array. Desk-checked — CuPy is not installed
  here — and the module docstring now states that the prescription walk
  host-round-trips and is therefore not `jit`/`vmap`/`grad`-traceable.

Files: `lumenairy/propagators/hfpi.py`,
`lumenairy/propagators/vectorial_hfpi.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK14K23GuardsAndCaps`
(5 tests, including the counter-pin that the default stratification is
unaffected).

---

### Added -- propagators/system: warn when a Fresnel / SAS leg's resample-back CROPS the field (K6, P2)

After `fresnel_propagate` / `scalable_angular_spectrum_propagate` the
field lives at `dx_new` over `N*dx_new`; `propagate_through_system` then
resamples it onto `N*current_dx`, and when `dx_new > current_dx` —
the common diverging-beam case — everything outside the central window is
discarded by `map_coordinates(mode='constant', cval=0)`, silently.
Measured (grid-filling top-hat, N = 512, dx = 2 um, z = 5 mm,
`dx_new/dx = 1.5454`): `P_out/P_in` = 0.998990 (asm, band limit,
expected) vs 0.996685 (fresnel) and 0.950689 (sas); the Fresnel step
itself conserves power to 1.000000 and the retained window holds 0.996980
of it, so essentially all of the Fresnel loss is the crop. A
`RuntimeWarning` now names the retained fraction; values are unchanged.

Files: `lumenairy/propagators/system.py`.

---

### Changed -- propagators/mft: `resample_field`'s documented MTF is the measured one (K6, P2)

The docstring rated the cubic-spline resampler at "< 0.1 % when features
are sampled at >= 4 pixels". It interpolates the real and imaginary parts
SEPARATELY, so a near-Nyquist complex carrier is attenuated; the measured
power ratios are 0.999549 / 0.990621 / 0.931504 / 0.718458 at 0.10 / 0.20
/ 0.30 / 0.40 cycles per pixel, which brackets the "4 pixels per feature"
case (0.25 cyc/px) between **0.9 % and 6.8 %** — at least an order of
magnitude worse than documented. The docstring now carries the table, the
reason it bites hardest on the single-FFT Fresnel output (whose residual
chirp sits at exactly Nyquist at the grid edge by construction), the
pointer to the band-limited `angular_spectrum_propagate_mft(z=0)`
alternative, and the fact that there is no anti-alias low-pass on
downsampling.

Files: `lumenairy/propagators/mft.py`.

---

### Changed -- propagators/hf, mhs: module docstrings describe the code (K15, K24, P3)

* **`hf.py`** advertised "the direct Huygens-Fresnel diffraction integral
  with the Van Vleck density correction", and named
  `propagate_huygens_fresnel` "the recommended entry point for new code"
  — that entry point is a three-line delegation to
  `rayleigh_sommerfeld_propagate` with no Van Vleck factor, and its own
  summary line claimed "the standard `1/(i lambda z)` Van Vleck factor"
  for a kernel whose leading term is `cos(theta)/(i lambda r)`. The
  docstring now separates the module's two unrelated halves and says
  which one each entry point is.
* **`mhs.py`** opened by describing rays propagating geometrically within
  subdomains and "a Huygens-surface integral" converting a ray bundle to
  a field at each surface. There is no ray bundle, no Huygens-surface
  integral and no ray tracing anywhere in the file, and
  `HuygensSurface` is flat-only. The paragraph is demoted to background
  and the module is described as what it is — a composition framework
  over delegated propagators.
* **`finite_diff_step`'s documented accuracy** ("−2.53e-8 at the 1e-6
  default") was measured on an exact-quadratic OPL where the 4th-order
  truncation term vanishes identically. On a spherical OPL the default
  gives **+2.4e-7** and the optimum moves a decade to `h = 1e-5`
  (−3.8e-8). Both are ~4 decades below the quadrature's own ~1e-3 floor,
  so this is documentation only — but the numbers now say which OPL they
  were measured on.
* **`__all__` integrity**: `propagate_huygens_fresnel` and
  `propagate_hfpi` — the documented canonical entry points — were absent
  from their modules' `__all__` (reachable throughout as
  `lumenairy.propagate_*`, so an integrity gap rather than a breakage).
  Added.

Files: `lumenairy/propagators/hf.py`, `lumenairy/propagators/hfpi.py`,
`lumenairy/propagators/mhs.py`.

---

### Fixed -- propagators/mhs: `MhsPipeline._validate` compares `centre` (K14, K24, P3)

The chain check compared `z` / `Ny` / `Nx` / `dx` but not `centre`, so
two surfaces at identical z/N/dx whose centres differ were accepted as
the same plane and the transverse jump was silently discarded (measured
at 50 um and at 1 mm) — while `HuygensSurface.grid()` and
`aperture_subdomain` both honour `centre`, so the field really is on a
different coordinate system either side. The error message now prints
every field of both surfaces.

Files: `lumenairy/propagators/mhs.py`.
Tests: `tests/unit/test_audit2609_a5_propagators.py::TestK14MhsSurfaceCentre`.

---

### Changed -- propagators/mhs: the two subdomain constructors' differing default method is documented (K24, P3)

`MhsPipeline.from_prescription` defaults to `method='gbd'` and
`prescription_subdomain` to `method='maslov'` — two constructors for the
same subdomain, two different physics models unless the method is named,
and `'maslov'` is the one that needs the square-grid guards and the
post-hoc resample. Neither default is changed (both are long-standing
public contracts); both docstrings now cross-reference the other and say
to name the method explicitly.

Files: `lumenairy/propagators/mhs.py`.

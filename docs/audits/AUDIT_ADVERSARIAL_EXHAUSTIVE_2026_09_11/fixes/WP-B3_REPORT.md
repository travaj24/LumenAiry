# WP-B3 — propagator kernels: the four designs WP-A5 deferred (K6, K9, K13, K22)

Branch `audit-fixes-2026-09`, base HEAD `81d5b586` (= release 5.46.0).
Scope: WP-A5's report §6 items 1–4, all four implemented.
Date of every measurement below: **2026-09-13**, this host, every Python
run under `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

---

## 1. Summary

| # | item | status | files : lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|---|
| 1 | **K13** — the prescription walk gets an explicit output plane (`z_output`), and `normalisation='auto'` | **landed** | `hfpi.py:1451`, `:1456`, `:1605`, `:1643–1678`, `:1790–1803`, `:1807`; `_walk_legs_are_free_space` `hfpi.py:1075` | `TestK13PrescriptionWalkOutputPlane` (10) | band-limited ASM through a flat-optics prescription; the oracle-free open-stop transparency property | scale vs ASM **1.16e-7 → 0.9906** (six seeds, max deviation 0.0149); flat in output pixel area (0.9906 / 0.9637 over ×1 / ×2) and in path count (0.9841 / 0.9906 / 0.9899 at 0.5 / 2 / 8 M) |
| 2 | **K22** — `sampler='jittered' \| 'sobol'` for the stratified source draw | **landed** | `hfpi.py:1141` (`_sobol_cube_draw`), `:1196` (`_jittered_cube_draw`), `:1314`, `:1387`, `:1453`, `:1697` | `TestK22SobolSampler` (7) | band-limited ASM, as the common reference both samplers converge to | **convergence order p = 0.537 (jittered) vs 0.557 (sobol)** — both Monte-Carlo, not the textbook `O(N^-1)`; error ratio 1.00–1.13× in Sobol's favour; `n_paths` 14 641 → **16 384 exactly** |
| 3 | **K9** (second half) — `kernel='spatial-integrated'`, the Shen–Wang pixel-integrated RS kernel | **landed, NOT the default (measured, see §3.1–3.2)** | `rs.py:87`, `:268`, `:679`, `:783`, `:800–812`, `:822–826`; `hf.py:256`, `:369` (docstrings) | `TestK9PixelIntegratedRsKernel` (9) | my own super-sampled continuum RS-I double quadrature of the staircase aperture; the exact on-axis closed form; the exact Hankel quadrature | vs the staircase oracle **4.7464e-3 → 4.5979e-6** (1032×, and *below* the oracle's own 1.3794e-5 floor); on a smooth Gaussian the ranking **reverses by 4–5 decades**, which is why `'auto'` keeps the point kernel |
| 4 | **K6** (second half) — `resample_field(method='chirpz')` | **landed; the call-site switch is specified, not made** | `mft.py:95`, `:127`, `:489`, `:550`, `:688`, `:742` | `TestK6ChirpZResampler` (7) | the Dirichlet-kernel interpolant as an explicit double sum; `map_coordinates` driven directly for the default leg | MTF at 0.40 cyc/px **0.717718 → 1.000000**; at 0.30 **0.930135 → 1.000000**; default leg **byte-identical** |

Everything whose default did not move is proved byte-identical against
the HEAD modules, not asserted — §6.

**Three of the four designs did not survive contact with measurement in
the shape the deferral described them,** and §3 is where each is
restated with the numbers that moved it:

* Item 3 was deferred with "the gain is the `O(dx) → O(dx^2)`
  convergence rate the audit measured for hard-aperture inputs". It is
  not — §3.2 shows the lever is the aperture's edge, for either kernel —
  and §3.1 shows the new kernel is four to five decades WORSE on a
  sampled smooth field, so it ships opt-in and `'auto'` never selects it.
* Item 2 was deferred with the note that the QMC advantage "should be
  checked before advertising it". It was: both samplers measure
  `O(N^-1/2)`, so the changelog advertises an exact path count and a
  1.00–1.13× constant, and no rate.
* Item 1's design said an output plane makes `normalisation='physical'`
  "well-defined … and can become the default here too". An output plane
  is necessary and not sufficient: §3.3 measures 4879× through a thin
  lens, so `'auto'` requires free-space legs as well.

Item 4 landed as designed; what did not survive was the assumption that
the two call sites could then simply be switched (§5.1).

---

## 2. Per item

### 2.1 K13 — `propagate_hfpi_through_prescription` gets an output plane

**What was wrong.** The walk binned the bundle wherever the surface list
left it. A diffracting last surface has just re-emitted every path, so
the final leg has length zero and `_binning_jacobian`'s
`r/(dx_out^2 cos theta_out)` — the factor that makes HFPI's amplitudes
the Huygens–Fresnel integral's — has no `r`. WP-A5 left the walk
defaulting to `normalisation='legacy'` with a warning saying the
amplitudes are not photometric.

**What landed.** `z_output: float | None = None`. When given, the walk
closes with `propagate_to_plane(paths, z_target=z_output, ...)` in the
medium the prescription puts after its last surface
(`get_glass_index(surfaces[-1].glass_after, wavelength)`), so an immersed
image space is handled. A `'MIRROR'` marker is normalised by
`surfaces_from_prescription` to the surface's own `glass_before` —
reflection does not change the medium — and the fold shows up as a
direction reversal, so `z_output` then has to sit on the side the
reflected paths travel toward (pinned by a test).

`normalisation` gains `'auto'` and becomes the default. It resolves to
`'physical'` iff **both** conditions the estimator needs hold, and to
`'legacy'` (with the existing warning) otherwise:

1. `z_output` gives the walk a plane to close on, and
2. every leg of the walk is free space
   (`_walk_legs_are_free_space`, `hfpi.py:1075`).

**Why condition 2 exists, with the number.** `_binning_jacobian` and
`_reemission_measure` convert emitted solid angle to landed area with the
free-space ray-tube relation `dS = r^2 dOmega / cos(theta)`. That is a
statement about a straight line in a uniform medium. Through an element
with power the system's own Jacobian applies instead, and the per-path
factor does not know. Measured through a 19.41 mm N-BK7 thin singlet
(R = ±20 mm, d = 10 µm, object 60 mm, 2 M paths, 0.02 rad cone,
`diffracting_surfaces=[]`) against ASM + thin-lens phase + ASM on the
same geometry:

| output plane | `normalisation='physical'` power / ASM power | centroid (walk vs ASM) | r84 (walk vs ASM) |
|---|---|---|---|
| image plane, 28.70 mm | **4879.08×** | (0.03, 0.06) vs (−0.07, −0.07) µm | 25.25 vs 31.93 µm |
| 14.35 mm | **13.886×** | (−0.48, −1.37) vs (−0.02, −0.02) µm | 211.66 vs 90.54 µm |

Shape is in the right ballpark at the image plane; the absolute scale is
wrong by 3.7 decades. Making `'physical'` the default for *any* walk with
an output plane — which is the literal reading of the deferred design —
would have shipped that as the default answer. `'auto'` therefore refuses
it there, and an explicit `normalisation='physical'` is honoured but
warns with these numbers (re-normalising a usable shape against a
reference is a legitimate workflow; being handed a 4879× amplitude
silently is not).

**Oracle and result.** Band-limited ASM through a flat-optics
prescription — one plane, index-matched, in air — which is exactly the
regime where the free-space Jacobian is the right one. 32×32 at
dx = 4 µm, w0 = 12 µm, z = 2 mm, cone 0.05 rad, unbiased least-squares
complex scale over the pixels above 5 % of the peak, six seeds at 2 M
paths:

| output pitch | scales (six seeds) | mean | max \|s−1\| |
|---|---|---|---|
| ×1 | 0.9891 0.9970 0.9857 0.9914 0.9955 0.9851 | 0.9906 | 0.0149 |
| ×2 | 0.9649 0.9689 0.9559 0.9648 0.9683 0.9595 | 0.9637 | 0.0441 |
| ×4 | 0.8670 0.8768 0.8653 0.8724 0.8724 0.8634 | 0.8695 | 0.1366 |

Flat in output pixel area over a 4× range — the K13 property itself —
and flat in path count: 0.9841 / 0.9906 / 0.9899 at 0.5 M / 2 M / 8 M
(three-seed means). The ×4 residual is the coarse bin averaging a field
the reference point-samples, not the Jacobian.

Second, oracle-free: an *open* stop must be transparent, so a two-leg
walk equals a one-leg walk over the same total distance. Four seeds at
2 M paths, cone 0.05 rad: two-leg **1.0428 ± 0.0499**, one-leg
**0.9908 ± 0.0041**, ratio **1.0524**.

**Fail-before.** The same call without `z_output`: scale vs ASM
**1.16e-7** (seven decades below), and the "NOT photometric" warning
still fires. Forcing `normalisation='physical'` there raises
`accumulate_to_grid: normalisation='physical' needs each path's
geometric distance …` rather than returning zeros. Both are asserted.

**Structural pin.** The closed walk through a flat index-matched plane
*in N-BK7* equals the public three-call sequence
`init_paths_stratified(rng=_spawn_rng(rng, 0))` →
`propagate_to_plane(n_medium=1.515082)` → `accumulate_to_grid` to a
relative L2 below 1e-12. Reading the medium as air instead would move
the phase by `k (n−1) z` = 1.02e4 rad, so this pins the `glass_after`
lookup as well as the hop itself.

**Cost** (medians of five interleaved runs, flat prescription,
32×32 output): 60.89 → **80.56 ms** at 131 072 paths (1.32×) and
679.11 → **806.06 ms** at 1 048 576 (1.19×). The closing hop is one
`propagate_to_plane`; the ratio is that big only because the walk it is
added to is short.

**Observation, not an item.** The cascaded (two-emission) estimator's
variance grows sharply with the emission cone: the transparency ratio
above is 1.0524 at 0.05 rad but **2.285** (two-leg 2.238 ± 1.051) at
0.12 rad and **15.29** (14.85 ± 5.68) at 0.25 rad, with the one-leg arm
steady at 0.99 / 0.98 / 0.97 throughout. The `r_in` factor
`_reemission_measure` carries makes the per-path weight heavy-tailed, so
the mean converges slowly. Recorded in §8.

### 2.2 K22 — the Sobol sampler

**What landed.** `init_paths_stratified(sampler={'jittered','sobol'})`,
default `'jittered'`, threaded through
`propagate_hfpi_through_prescription`. `'sobol'` draws
`scipy.stats.qmc.Sobol(d=4, scramble=True, seed=…).random(n_paths)` over
the same `(pixel_x, pixel_y, cos theta, phi)` cube, with the
measure-preserving mapping the jittered sampler uses (affine in
`cos theta` over `[cos_max, 1]`, so equal solid angle per unit
coordinate). The scramble seed is drawn from the same `RandomState`, so
the bundle stays a pure function of `rng` on every backend. Owen
scrambling is **on**: an unscrambled Sobol sequence is a fixed point set
whose error is a deterministic bias with no way to measure it.

`n_paths` is honoured **exactly** — there is no stratification grid to
round onto. The jittered 4th-root rule gives 14 641 for a requested
16 384; Sobol gives 16 384.

To keep the request unambiguous rather than silently ignored,
`sampler='sobol'` with an explicit `n_strata_xy` / `n_strata_dir` raises,
`sampling != 'stratified'` with a non-default sampler raises, and a
non-power-of-two `n_paths` warns (a Sobol sequence is balanced only on
its `2**m` prefixes; the warning names the two nearest powers).

**The measurement, which is the deliverable.** Reference: band-limited
ASM on the same 32×32 / dx = 4 µm / w0 = 12 µm / z = 2 mm / 0.05 rad
geometry, over the pixels above 5 % of the peak, six seeds per point.
Two error measures, because a convergence RATE is a statement about the
second one:

* **`err_total`** — relative L2 of the SEED-MEAN estimate against ASM
  (bias + what the averaging leaves of the noise);
* **`err_noise`** — RMS scatter of the individual estimates about their
  own seed-mean, relative to the reference norm. This is the
  estimator's own error and it does not care whether the reference is
  exact.

| n_paths requested | jittered actual | jit `err_noise` | jit `err_total` | sobol actual | sob `err_noise` | sob `err_total` | noise ratio jit/sob |
|---|---|---|---|---|---|---|---|
| 16 384 | 14 641 | 3.9776 | 1.6443 | 16 384 | 3.7742 | 1.5129 | 1.054 |
| 32 768 | 28 561 | 2.8445 | 1.1566 | 32 768 | 2.6601 | 1.0728 | 1.069 |
| 65 536 | 65 536 | 1.8530 | 0.74667 | 65 536 | 1.8291 | 0.74797 | 1.013 |
| 131 072 | 130 321 | 1.3033 | 0.54585 | 131 072 | 1.2646 | 0.52084 | 1.031 |
| 262 144 | 262 144 | 0.92283 | 0.37553 | 262 144 | 0.85456 | 0.33978 | 1.080 |
| 524 288 | 524 288 | 0.63563 | 0.26235 | 524 288 | 0.56358 | 0.23140 | 1.128 |
| 1 048 576 | 1 048 576 | 0.41198 | 0.16946 | 1 048 576 | 0.39704 | 0.16877 | 1.038 |

Fitting `err ~ N^-p` over the last five points:

| sampler | p (`err_total`) | p (`err_noise`) |
|---|---|---|
| jittered | 0.533 | **0.537** |
| sobol | 0.547 | **0.557** |

The two agree, which is itself the check that the reference is not the
thing being measured.

**Both are Monte-Carlo.** The textbook `O(N^-1)` does not appear, and the
seed-mean scale is 0.9867–1.0055 (jittered) and 0.9899–1.0061 (sobol)
throughout, so the estimator is unbiased in both cases. The 16× error
ratio measured over five *independent* three-seed groups is
4.349 / 4.522 / 4.533 / 4.417 / 4.492 (jittered) and 4.668 / 4.648 /
4.719 / 4.628 / 4.631 (sobol), against 4.0 for `O(N^-1/2)` and 16.0 for
`O(N^-1)`. The integrand has hard edges — the cone cut, the output-pixel
bin, any aperture — and that is what the numbers say.

So the honest claim in the changelog is: **`'sobol'` buys an exact path
count and a 1.00–1.13× constant, not a convergence rate.**

**Cost** (medians of seven interleaved runs): the draw itself is
0.94× / 0.91× / **0.80×** the jittered one at 2^14 / 2^17 / 2^20 paths
(157.45 ms against 195.92 ms at 2^20); a whole init + propagate + bin
estimate is 1.25× at 2^17 and 0.94× at 2^20.

### 2.3 K9 second half — the pixel-integrated RS kernel

**What landed.** `kernel='spatial-integrated'` on
`rayleigh_sommerfeld_propagate`. `_rs_pixel_integrated_kernel`
(`rs.py:268`) returns
`Int_{pixel} h(x, y, z) dx dy` on the padded grid by a
`_RS_PIXEL_QUAD_NODES = 6`-node tensor Gauss-Legendre rule per pixel.
Because `h` depends on `x` and `y` only through `x**2 + y**2` the build
runs on one quadrant (`Ny//2 + 1` by `Nx//2 + 1`, a quarter of the padded
grid) and is gathered out by the index map `|m - Ny2//2|`; `x**2` is
bit-identical for `x` and `−x`, so the folded build equals an unfolded
one to 2.1e-16 (round-off from the mirrored summation order; asserted in
the test file against an unfolded rule written there). Its own H-cache
tag is `'RS_INT'`, so it cannot be handed `'spatial'`'s array.

Node count chosen from the measured floor at the WORST legal geometry
`z = 2 N dx^2 / lambda`, where the kernel's phase sweeps its full `pi`
across a pixel (relative L2 against a 14-node build):

| grid | n=3 | n=4 | n=5 | **n=6** | n=7 |
|---|---|---|---|---|---|
| N = 64, dx = 2 µm | 1.96e-4 | 1.85e-6 | 1.12e-8 | **4.73e-11** | 1.81e-13 |
| N = 128, dx = 1 µm | 1.48e-4 | 1.28e-6 | 7.06e-9 | **2.72e-11** | 1.03e-13 |
| N = 128, dx = 2 µm | 1.96e-4 | 1.84e-6 | 1.11e-8 | **4.64e-11** | 2.56e-13 |
| N = 256, dx = 1 µm | 1.48e-4 | 1.27e-6 | 7.04e-9 | **2.71e-11** | 1.57e-13 |

**Oracle.** The brute-force RS quadrature's *method*, written from
scratch (`_rs_staircase_quadrature`, committed in the test file; the
`repro/` script was not imported or read for its code): the exact
continuum RS-I integral of the
piecewise-constant field the array represents, as a sum over LIT pixels
of an `S × S` super-sampled MIDPOINT rule — deliberately a different
quadrature from the module's Gauss-Legendre one, by direct summation at
four output points, no FFT, no padding, no library call. Its own floor is
second order in `1/S` and is read off by running it at `S` and `2S`.

λ = 633 nm, circular aperture a = 100 µm, window 512 µm, **z = 16 mm**
(above the alias threshold of every grid quoted, so both spatial kernels
are legal on all of them; the Fresnel number is 0.99, which puts the
closed-form on-axis value at order 2 rather than near a zero):

| grid | S | oracle floor | `'spatial-integrated'` | `'spatial'` | ratio |
|---|---|---|---|---|---|
| N = 64, dx = 8 µm | 16 | 1.3794e-5 | **4.5979e-6** | 4.7464e-3 | 1032× |
| N = 64, dx = 8 µm | 32 | 3.4485e-6 | **1.1495e-6** | 4.7499e-3 | 4132× |
| N = 128, dx = 4 µm | 16 | 3.3558e-6 | **1.1186e-6** | 1.1465e-3 | 1025× |
| N = 128, dx = 4 µm | 32 | 8.3895e-7 | **2.7965e-7** | 1.1473e-3 | 4103× |

The integrated kernel sits *below* the oracle's own floor and divides by
four every time `S` doubles: that is the ORACLE converging onto it, which
is what "exact" looks like when the only available reference is itself
approximate. The point-sampled kernel does not move with `S` at all.

### 2.4 K6 second half — the chirp-Z resampler

**What landed.** `resample_field(..., method={'spline','chirpz'})`,
keyword-only, default `'spline'` = the historical cubic
`map_coordinates` leg, byte-identical. `'chirpz'`
(`_resample_field_chirpz`, `mft.py:489`) transforms to the centred
spectrum and inverse-transforms it straight onto the output grid with
`_bluestein_centred_2d`, folding the odd-`N` half-pixel origin offset
into the output centre exactly as `angular_spectrum_propagate_mft` does.
No transfer function is applied, so — unlike
`angular_spectrum_propagate_mft(z=0, …)`, which the deferral suggested as
the drop-in — the evanescent set is not zeroed and a sub-`lambda/2` pitch
is not silently low-passed.

**Oracles.** (a) The Dirichlet-kernel interpolant as an explicit double
sum over the centred DFT bins, written out in the test file: agreement
1e-14 class, bar 1e-12. (b) `map_coordinates` driven directly with the
documented coordinate map, for the default leg: BYTE identity. (c)
Parseval on a pure carrier, for the MTF.

| carrier (cyc/px) | px/cycle | spline (×0.5) | chirpz (×0.5) | spline (×1.5) | chirpz (×1.5) |
|---|---|---|---|---|---|
| 0.00 | ∞ | 0.999999 | **1.000000** | 0.999999 | **1.000000** |
| 0.05 | 20 | 0.999962 | **1.000000** | 0.999957 | **1.000000** |
| 0.10 | 10 | 0.999497 | **1.000000** | 0.999440 | **1.000000** |
| 0.20 | 5 | 0.990258 | **1.000000** | 0.989332 | **1.000000** |
| 0.30 | 3.33 | 0.930135 | **1.000000** | 0.924956 | **1.000000** |
| 0.40 | 2.5 | 0.717718 | **1.000000** | 0.700854 | **1.000000** |

`dx_out == dx_in` returns the input to 9.5e-15 relative (the spline leg's
own identity is 2.3e-16 — it is a genuine identity there, the chirp-Z leg
is a transform pair).

**The property that constrains the call sites.** The chirp-Z
reconstruction is periodic with period `N_in*dx_in`; a window wider than
that returns replicas, not the zeros the spline pads with. Measured for a
2× window: power ratio exactly **4.000000** (a 2×2 tiling) against the
spline's 1.000000. That case warns, reusing the MFT family's
faithful-zone diagnostic; `_warn_mft_output_window` grew an optional
per-axis `N_out_y` so the non-square extent-preserving default can use it
(every existing caller is square, and the warning text is proved
unchanged for them).

**Cost** (medians of seven interleaved runs, ms): 1.88 → 2.01 (N = 64,
×0.5), 8.19 → 8.76 (N = 128), 32.70 → 48.18 (N = 256), 154.52 → 185.12
(N = 512), 25.55 → 45.42 (N = 512, ×1.5). I.e. 1.07×–1.78×.

---

## 3. The three findings that contradict the deferred designs

### 3.1 `'spatial-integrated'` must NOT become the default

The deferral asked for the convergence table and for a recommendation.
The recommendation is **no**, and it is not close.

Against an adequately sampled SMOOTH input the ranking reverses by four
to five decades. Relative L2 against the exact Hankel angular-spectrum
quadrature of a Gaussian, six geometries, all legal for both kernels:

| grid | z | `'spatial'` | `'spatial-integrated'` |
|---|---|---|---|
| N = 128, dx = 1 µm, w0 = 6 µm | 3 mm | **4.42e-8** | 1.24e-3 |
| N = 128, dx = 1 µm, w0 = 6 µm | 0.5 mm | **6.63e-8** | 3.27e-3 |
| N = 128, dx = 2 µm, w0 = 12 µm | 1.7 mm | **3.23e-8** | 3.27e-3 |
| N = 64, dx = 2 µm, w0 = 6 µm | 0.9 mm | **6.46e-8** | 1.29e-2 |
| N = 256, dx = 0.5 µm, w0 = 6 µm | 0.25 mm | **6.35e-8** | 8.18e-4 |
| N = 256, dx = 1 µm, w0 = 12 µm | 1 mm | **3.34e-8** | 8.18e-4 |

The gap is exactly the difference between the Gaussian and its own
staircase (the step between the two kernels equals the integrated
kernel's error to every digit printed). The two kernels are not better
and worse versions of one operator; they are the operator applied under
two different readings of what a sample means, and the library's fields
are point samples of continuous functions almost everywhere. `'auto'`
keeps `'spatial'`.

### 3.2 The convergence rate the deferral promised is the APERTURE, not the kernel

The deferral's stated gain was "the `O(dx) → O(dx^2)` convergence rate
the audit measured for hard-aperture inputs (2.86e-2 → 1.08e-3 over
N = 256…2048)". Measured, the integrated kernel does not deliver it,
because that error is the pixel-centre-indicator representation of a
circle, which neither kernel touches. On-axis relative error against the
closed form `U = e^{ikz} - (z/r_a) e^{ik r_a}`, λ = 633 nm, a = 100 µm,
window 512 µm, z = 16 mm:

| N | dx [µm] | stair + point | stair + integrated | grey + point | grey + integrated |
|---|---|---|---|---|---|
| 128 | 4.000 | 8.3008e-3 | 8.4184e-3 | 1.4045e-3 | 2.6828e-3 |
| 256 | 2.000 | 3.3548e-3 | 3.3707e-3 | 3.4263e-4 | 6.6308e-4 |
| 512 | 1.000 | 3.4207e-4 | 3.5238e-4 | 8.4251e-5 | 1.6429e-4 |
| 1024 | 0.500 | 5.2718e-4 | 5.2758e-4 | 2.0677e-5 | 4.0724e-5 |

order between successive rows:

| arm | 128→256 | 256→512 | 512→1024 |
|---|---|---|---|
| stair + point | 1.307 | 3.294 | **−0.624** |
| stair + integrated | 1.321 | 3.258 | **−0.582** |
| **grey + point** | **2.035** | **2.024** | **2.027** |
| **grey + integrated** | **2.016** | **2.013** | **2.012** |

"grey" is the aperture as its exact pixel-AREA average. The lever that
restores second order is the input's edge, for **either** kernel — 25×
at N = 1024 — and the kernel choice then moves the constant by ~2×, in
the point-sampled kernel's favour. The staircase arms are not slowly
converging so much as non-monotone: a circle's staircase area error does
not shrink smoothly, which is why the audit's 2.86e-2 → 1.08e-3 looked
like first order.

The same lever, measured the same way on `hf.py`'s OPL quadrature (one
on-axis output point, spherical `Phi`, so the `O(N_in^2 N_out^2)` cost is
`N_in^2`): indicator 2.7708e-2 / 1.1120e-2 / 1.1509e-3 / 1.7601e-3
(orders 1.32 / 3.27 / **−0.61**) against area-average 1.4762e-2 /
3.6171e-3 / 8.9158e-4 / 2.2423e-4 (**2.03 / 2.02 / 1.99**), 7.9× better
by N = 1024. That table is now in
`propagate_huygens_fresnel_with_opl_callable`'s docstring, under a
"What sets the accuracy floor" heading, because it is a free accuracy
gain a caller can take today.

### 3.3 `normalisation='physical'` is not definable by an output plane alone

See §2.1. `'auto'` requires free-space legs as well, because the
free-space ray-tube Jacobian is off by 4879× at a thin lens's image
plane.

---

## 4. Files touched

| file | what |
|---|---|
| `lumenairy/propagators/rs.py` | `_RS_PIXEL_QUAD_NODES` (`:87`); `_rs_pixel_integrated_kernel` (`:268`); `kernel` vocabulary (`:679`) and its docstring entry; alias guard over both spatial kernels (`:783`); `'RS_INT'` cache tag (`:800–812`); the build branch (`:822–826`); reference [2] restated |
| `lumenairy/propagators/mft.py` | `_warn_mft_output_window(N_out_y=…)` (`:95`, `:127`); `_resample_field_chirpz` (`:489`); `resample_field(method=…)` (`:550`, `:688`, `:742`) and its docstring |
| `lumenairy/propagators/hfpi.py` | `_walk_legs_are_free_space` (`:1075`); `_sobol_cube_draw` (`:1141`); `_jittered_cube_draw` (`:1196`, the existing block moved verbatim); `init_paths_stratified(sampler=…)` (`:1314`, `:1387`); `propagate_hfpi_through_prescription(z_output=…, sampler=…, normalisation='auto')` (`:1451–1456`, `:1605–1678`, `:1697`, `:1790–1807`) |
| `lumenairy/propagators/hf.py` | **docstrings only** — the `kernel=` pass-through (`:256`) and the "What sets the accuracy floor" table (`:369`) |
| `docs/history/lumenairy.propagators.rs.md` | re-recorded |
| `docs/history/lumenairy.propagators.mft.md` | re-recorded |
| `docs/history/lumenairy.propagators.hfpi.md` | re-recorded (**two** `re_recorded:` lines — the second is the refusal message catching up with the docstring, recorded separately rather than silently folded into the first) |
| `tests/unit/test_audit2609_b3_propagator_kernels.py` | new, 33 tests |
| `docs/audits/.../fixes/WP-B3_REPORT.md`, `WP-B3_CHANGELOG.md` | this report and the release text |

`docs/history/lumenairy.propagators.hf.md` is **unchanged**, and that is
the recorder's own verdict rather than an omission: the write was
attempted —

```
python scripts/record_history_fingerprints.py lumenairy/propagators/hf.py \
    --reason "WP-B3 (audit K9 second half): docstring-only edit -- ..."
→ OK    lumenairy.propagators.hf.md    lumenairy/propagators/hf.py
```

— and the script short-circuits with `OK` and writes nothing when a
module has not drifted. Both fingerprints strip docstrings, and the
hf.py edit is docstrings only, so neither moved. Appending a
`re_recorded:` line there would date a baseline move that did not
happen, which is the one thing the header exists to keep honest.
`--check` lists all four of my documents `OK`.

No file outside this list was modified. `system.py`, `_lens_real.py`,
`lenses_maslov.py`, `carrier.py` and `asymptotic*.py` were read only.

---

## 5. Requested changes outside my ownership

### 5.1 `lumenairy/propagators/system.py` — the `fresnel` and `sas` resample-back legs (K6)

**Do not switch these to `method='chirpz'` unconditionally.** The switch
is only safe in one direction, and on the Fresnel leg there is a better
change available. Measured on the K6 fixture (grid-filling top-hat of
radius `0.42*N*dx`, λ = 633 nm), against
`fresnel_propagate_mft(E, z, λ, dx_in=current_dx, dx_out=current_dx,
N_out=N)` — the SAME Fresnel integral evaluated directly on the chain
grid, so it is the field the leg is trying to produce and it does not
resample at all:

| fixture | `dx_new/dx` | spline relL2 vs direct | chirpz relL2 vs direct | spline P/P_in | chirpz P/P_in | direct P/P_in |
|---|---|---|---|---|---|---|
| N = 256, z = 5 mm, top-hat | 3.0908 | 4.3589e-2 | 4.9204e-2 | 0.986005 | **0.986947** | 0.986945 |
| N = 256, z = 2 mm, top-hat | 1.2363 | 4.8591e-2 | 5.3308e-2 | 0.995421 | **0.996056** | 0.996072 |
| N = 512, z = 5 mm, top-hat | 1.5454 | 2.8108e-2 | 3.1474e-2 | 0.996685 | **0.996993** | 0.996992 |
| N = 256, z = 5 mm, contained Gaussian | 3.0908 | 2.2158e-6 | **1.5314e-8** | 0.999999 | 1.000000 | 1.000000 |
| N = 512, z = 5 mm, contained Gaussian | 1.5454 | 1.3336e-8 | **1.0611e-8** | 1.000000 | 1.000000 | 1.000000 |

Three readings:

1. **On a contained field the chirp-Z leg is a clean win** — 145× more
   accurate at `dx_new/dx = 3.09`, and it reproduces the direct
   evaluation's window power to ~1e-5 where the spline is 0.03–0.09 %
   low (that gap IS the MTF).
2. **On a grid-filling field neither resampler is the problem**: both
   sit 2.8e-2–5.3e-2 from the direct evaluation, i.e. the resample-back
   itself is a 3–5 % modelling error. The spline is marginally closer
   only because its roll-off happens to suppress content that the
   band-limited interpolant faithfully keeps.
3. **In the converging direction the chirp-Z leg is WRONG**, because the
   requested window then exceeds the reconstruction's period:

| sas fixture | `dx_new/dx` | spline P/P_in | chirpz P/P_in | chirpz warns |
|---|---|---|---|---|
| N = 256, z = 5 mm | 1.5454 | 0.986730 | 0.986951 | no |
| N = 512, z = 5 mm | **0.7727** | 0.950689 | **1.378837** | yes |
| N = 512, z = 2 mm | **0.3091** | 0.174014 | **1.820005** | yes |
| N = 256, z = 1 mm | **0.3091** | 0.173601 | **1.835050** | yes |

so the requested edits are:

**(a) `system.py:813–814`, the Fresnel leg — replace the
propagate-then-resample pair with a direct MFT evaluation.** Strictly
better on every fixture above and it removes the crop *and* the resample
in one step:

```python
            if prop_method == 'fresnel' and not has_tilt:
                _require_square_pitch(current_dx, current_dy, 'fresnel')
                # Evaluate the Fresnel integral straight onto the chain
                # grid instead of taking the natural output grid and
                # interpolating back onto it: measured 2.8e-2 .. 4.9e-2
                # relative L2 removed on a grid-filling field, and the
                # window power moves 0.996685 -> 0.996992 (the resample
                # MTF) at N = 512, dx = 2 um, z = 5 mm.
                from .mft import fresnel_propagate_mft
                E = fresnel_propagate_mft(
                    E, z, wavelength, current_dx, current_dx,
                    int(E_in.shape[-1]), dy_in=current_dy,
                    dy_out=current_dx)
```

(deleting the `if abs(dx_new - current_dx) > …` block that follows,
including its `_warn_system_resample_crop` call — there is no resample
left to crop, and `fresnel_propagate_mft`'s own faithful-zone warning
takes over, with period `lambda*|z|/dx_in`. The signature is checked:
`fresnel_propagate_mft(E_in, z, wavelength, dx_in, dx_out, N_out, *,
dy_in=None, dy_out=None, centre_out=(0,0), use_gpu=False)`, returning
the bare array. I have not applied this edit myself, and it moves the
`fresnel` leg's numbers on every call that resampled — that is the
point, but it wants its own regression pass.)

**(b) `system.py:834–835`, the SAS leg — keep `resample_field`, and gate
the method on the direction:**

```python
                    # K6: the band-limited resampler has unit MTF, but
                    # its reconstruction is periodic with period
                    # N*dx_new, so it is only usable when the chain
                    # window fits inside that -- i.e. when the pitch
                    # COARSENED.  In the converging direction it returns
                    # replicas (measured P_out/P_in 1.38 and 1.82 where
                    # the spline gives 0.95 and 0.17).
                    E, _ = resample_field(
                        E, dx_new, current_dx, N_out=E_in.shape[-1],
                        method=('chirpz' if dx_new >= current_dx
                                else 'spline'))
```

**(c)** If (a) is judged too large a change for this release, apply the
same gated form as (b) to the Fresnel leg at `system.py:813`.

Wherever a `resample_field` call survives — (b), and (c) if taken — the
existing `_warn_system_resample_crop` call stays with it: the crop is
still real, and on a grid-filling field it is the larger of the two
effects by two decades. Only (a) retires it, because only (a) removes
the resample.

### 5.2 `lumenairy/elements/_lens_real.py` — `_propagate_gap`'s `sas` and `fresnel` legs (K6)

Same physics, same gate. In `_propagate_gap` (the two lines currently
reading `E, _ = resample_field(E, dx_new, dx, N_out=E.shape[-1])`, found
at `:2767` (sas) and `:2772` (fresnel) in the tree I read — WP-B2 is
editing this file, so match on the text rather than the line number):

```python
        if abs(dx_new - dx) > dx * 1e-6:
            # K6: unit-MTF resampling where the window fits inside the
            # chirp-Z reconstruction's period (N*dx_new), the spline
            # where it does not.
            E, _ = resample_field(
                E, dx_new, dx, N_out=E.shape[-1],
                method='chirpz' if dx_new >= dx else 'spline')
```

Note this leg propagates **in glass**, so `lam_medium = wavelength/n`
makes `dx_new = lam_medium*z/(N*dx)` smaller by `n` than the air case —
i.e. it lands in the converging direction more often than `system.py`'s
does, which is exactly why the gate matters here.

### 5.3 `lumenairy/propagators/hf.py` — the OPL quadrature's pixel integral: NOT recommended

`propagate_huygens_fresnel_with_opl_callable` is the other place a
pixel-integrated kernel would apply, and it is the one place it should
not be built. `Phi` is an arbitrary caller-supplied callable, so there is
no analytic pixel integral; the contained generalisation is an
`n_g × n_g` sub-pixel Gauss-Legendre rule, which multiplies `opl_fn`'s
already-dominant cost by `n_g^2` (36× at the node count `rs.py` needs) on
a path the audit measured at 9.56 ms per output pixel at `N_in = 256`, i.e.
~10 minutes for one output plane today and ~6 hours after. It would also
need sub-pixel values of `E_in`, which only exist as an interpolation,
re-introducing the error it set out to remove.

What that function needed was the measurement in §3.2, and it has it in
its docstring. **Requested of nobody; recorded so the next reader does
not re-derive it.**

### 5.4 `lumenairy/propagators/dispatch.py` — nothing requested, and why

Checked rather than assumed: the `method='hfpi'` branch forwards
`**kwargs` verbatim to `propagate_hfpi_through_prescription`
(`dispatch.py:1152–1157`) and there is no keyword allow-list anywhere in
`propagate()`, so `propagate(method='hfpi', prescription=…,
z_output=…, sampler=…)` already reaches the new keywords. Verified by
call, and now pinned
(`TestK13PrescriptionWalkOutputPlane::test_both_new_keywords_reach_the_dispatcher`)
so a future allow-list there cannot break it silently. **No edit
requested.**

The gap that does exist is on the other side: the free-space HFPI entry
points (`propagate_hfpi`, `propagate_hfpi_freespace_aperture`) call
`init_paths_from_field`, not the stratified sampler, so they have no
`sampling` selector for `sampler` to hang off. Giving them one is a
design change to those functions — a `sampling='uniform'|'stratified'`
kwarg threaded to the right initialiser — not a pass-through, and it is
outside these four items. Recorded in §8.

---

## 6. Byte-identity proofs

Every path whose default did not move was compared against the **HEAD
(81d5b586) module itself**, loaded under
`lumenairy.propagators._baseline_<name>` so its relative imports resolve
against the live package, with only the module under test swapped
(the harness lives in this session's scratchpad, not in the tree —
nothing to look for under `tests/` or `docs/`). `tobytes()` comparison,
plus `shape` and `dtype`. Result: **ALL BYTE-IDENTICAL** — 49 array
comparisons, 2 warning-text comparisons and 1 identical-refusal
comparison, 52 in all, zero differences. Re-run on the final tree after
every edit in this work package had landed.

| surface | cases |
|---|---|
| `rayleigh_sommerfeld_propagate` | `kernel='auto'` at (64, 2 µm, 2 mm), (128, 1 µm, 3 mm), (128, 1 µm, 50 µm — the transfer branch), odd N = 65; `kernel='spatial'`; `kernel='transfer'`; `bandlimit=True`; `complex64`; anamorphic 64×48 with `dy = 2 dx` |
| `resample_field` | default leg at five (N, dx_in, dx_out, N_out, order) combinations including odd N = 65 and the exact no-op; returned `dx_out` too |
| `angular_spectrum_propagate_mft`, `fresnel_propagate_mft`, `fraunhofer_propagate_mft` | the `_warn_mft_output_window` edit; including a window that triggers the warning |
| `_warn_mft_output_window` | warning TEXT identical for square callers, two fixtures |
| `init_paths_stratified` | jittered default, all six `PathBundle` fields, at 8×8/1024, 16×16/4096, the K23 explicit-stratification cap case, and non-square 12×9/5000 |
| `propagate_hfpi_through_prescription` | the singlet walk at `sampling='stratified'` and `'uniform'`; the flat walk at explicit `normalisation='legacy'`; explicit `'physical'` without `z_output` raises the identical message in both |
| `propagate_hfpi_freespace_aperture` | untouched entry point |
| `propagate_huygens_fresnel_freespace` | bare pass-through and the resample leg (both returned values) |

---

## 7. Tests run

The whole set below was run TWICE: once as each item landed, and again
end to end on the final tree after every edit in this work package had
been made. The table reports the **final** run; where the first pass
gave a different number it is noted, and the only differences are other
work packages moving underneath (§7.1).

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b3_propagator_kernels.py` | **33 passed** | 59.39 s |
| `pytest tests/unit -k "a5 or hfpi or hf_ or rs_ or resample or mft or bluestein"` | **768 passed, 12 skipped** (the skips are PySide6, CuPy, JAX-x64 and the W5 host-specific digest set — all pre-existing) | 288.45 s |
| `pytest tests/unit/test_niche_d2_chain_multi.py tests/unit/test_audit2609_a25_carrier_focus_readout.py tests/unit/test_audit2609_a17_history_lint.py tests/unit/test_audit2609_a17_history_relocation.py` | 792 passed, **7 failed — every one of them in another work package's module** (§7.1; the first pass had 5, the extra 2 are a third engineer's `carrier.py` arriving) | 577.32 s |
| `pytest tests/unit -k "system or propagat or kernel or sobol"` (an extra sweep, not in the verification set: everything that reaches `propagate_through_system` and `resample_field`'s other callers) | **1373 passed, 9 skipped** | 298.77 s |
| `python validation/run_all.py test_propagation test_hfpi test_hf test_dispatch test_advanced_diffraction --quiet` | **ALL 5 files passed** (2.6 / 1.8 / 2.7 / 3.7 / 7.3 s) | 18.1 s |
| `ruff check` (whole repo, and separately `lumenairy/ tests/`) | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | all **four** of my documents `OK`; the DRIFT entries are `carrier.py`, `_lens_real.py` and `lenses_maslov.py`, none of them mine | — |
| the HEAD byte-identity harness (§6) | **ALL BYTE-IDENTICAL**, 52/52 | — |

The b3 file's own slowest tests: 18.5 s
(`test_the_scale_does_not_move_with_path_count`, 8 M paths × 3 seeds),
14.6 s (`test_an_open_stop_is_transparent`, 2 M × 4 seeds × 2 walks),
13.5 s (`test_the_point_kernel_is_the_right_one_for_a_sampled_smooth_field`,
the Hankel oracle). No test asserts a duration, and every duration in
this section is indicative only — three engineers were sharing this box
throughout.

### 7.1 Every failure is another work package's, named

All seven are in modules owned by the other concurrent Wave-4 engineers,
and none of my four modules appears in any of them:

* `test_audit2609_a17_history_lint::test_no_module_accumulates_more_version_history`
  — the narrative ratchet, grown in `lumenairy/elements/_lens_real.py`
  and `lumenairy/elements/lenses_maslov.py`. The counts move while those
  files are being edited: the pre-work baseline I took BEFORE touching
  anything already had `_lens_real.py` at `0 → 3`; the full run above
  caught it at `0 → 2` plus `lenses_maslov.py` at `1 → 3`; a re-check
  ten minutes later reads `lenses_maslov.py: 1 → 3` alone, `_lens_real.py`
  having been cleaned in between. That it was red before I started is
  the point.
* `test_audit2609_a17_history_relocation` ×6 — AST and token drift on
  `carrier`, `lumenairy.elements._lens_real` and
  `lumenairy.elements.lenses_maslov`: three modules whose owners have not
  re-recorded yet. My four documents pass, three re-recorded and one
  (`hf`) legitimately untouched.

One transient worth recording, because it makes the whole tree look red
for a reason unrelated to anyone's content: at 06:49 a concurrent edit
left `lumenairy/propagators/carrier.py` with a
`SyntaxError: f-string: expecting '}'`, which fails `import lumenairy`
and therefore collects nothing anywhere. It cleared at 07:23 and the
final run above is entirely after that. I have never opened that file.

---

## 8. Deferred / observed, not fixed

1. **The cascaded HFPI estimator's variance at wide cones.** The
   transparency ratio (two-leg over one-leg, which must be 1) reads
   1.0524 at a 0.05 rad emission cone, **2.285** at 0.12 rad and
   **15.29** at 0.25 rad, four seeds at 2 M paths each, while the
   one-leg arm stays at 0.99 / 0.98 / 0.97. The `r_in` factor
   `_reemission_measure` carries makes the per-path weight heavy-tailed,
   so the sample mean converges slowly and from above. This is a
   variance problem, not the bias V1 fixed; importance-sampling the
   re-emission toward the output aperture is the standard remedy and is
   the same lever the audit's own "alternatives" note proposed for the
   v5.31 guard. Out of scope here (it changes the free-space entry
   points too).
2. **`normalisation='physical'` through a powered prescription.** The
   correct per-path factor is the system's ray-tube Jacobian
   `|dA/dOmega|`, which the differential ray tracer
   (`raytrace/differential.py`) already computes for GBD. Threading it
   into `_binning_jacobian` would make the walk photometric through real
   optics; that is a genuine design change, with the 4879× measurement
   in §2.1 as its acceptance test.
3. **`'spatial-integrated'` on CuPy and JAX.** The build is written
   against `xp` and uses only `arange`/`meshgrid`/`sqrt`/`exp`/`abs` and
   advanced indexing, so it should run on both; CuPy is not installed
   here and the JAX session has x64 off, so both are **desk-checked
   only**. Under JAX without x64 the coordinate `arange(dtype=float64)`
   silently becomes float32, exactly as the existing point-sampled
   branch does — same pre-existing caveat, not a new one.
4. **`apply_aperture(edge='gray')` should probably be the default, and
   every propagator docstring should point at it.** §3.2 shows the single
   largest accuracy lever available to a caller of either RS spatial
   kernel or of the HF quadrature is feeding it a grey-edged aperture —
   25× at N = 1024, and the difference between second order and no
   reliable order at all. The library already builds one:
   `elements/elements.py::apply_aperture` takes
   `edge={'hard','gray'}, edge_samples=4`, added 2026-09-12, whose own
   docstring measures the transmitted-AREA error (0.386 % hard against
   0.044 % gray at `D/dx ~ 50`). What was missing is the connection: that
   area error is what sets the PROPAGATED field's convergence order, and
   nothing said so. `rs.py`'s and `hf.py`'s docstrings now point at it
   (this work package); the remaining questions — should `edge` default
   to `'gray'`, and should the other aperture builders
   (`hfpi.apply_aperture_diffraction`'s hard mask, the prescription
   walk's `semi_diameter` cut) grow the same option — are for whoever
   owns `elements/elements.py`, with §3.2's table as the case.
5. **`resample_field` on non-NumPy backends.** Both legs are host-only —
   the spline via `scipy.ndimage`, the chirp-Z via the `_fft2`/`_ifft2`
   host FFT. Unchanged by this work package, recorded because the new
   leg could have been backend-generic and is not.
6. **`sampler` on the free-space HFPI entry points.** `propagate_hfpi`
   and `propagate_hfpi_freespace_aperture` call
   `init_paths_from_field`, so they have no stratified cube for a
   sampler to place points in and no `sampling` selector to hang one
   off. The prescription walk has both, which is why `sampler` landed
   there. Extending it means giving the free-space pair a
   `sampling='uniform'|'stratified'` kwarg and routing the initialiser
   — a design change to two public functions, not a pass-through. See
   §5.4 for what IS reachable today.

# WP-B3 changelog text — propagator kernels, the four deferred designs (K6, K9, K13, K22)

Release text for **5.47.0**, in the voice of
`fixes/WP-A5_CHANGELOG.md`, for the orchestrator to assemble. Audit:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`; the designs
are WP-A5's report §6 items 1–4. Every default in this work package is
**byte-identical** to 5.46.0 — each item is a new keyword whose default
value selects the old path, proved against the HEAD modules
(81d5b586) rather than asserted.

---

### Added -- propagators/hfpi: the prescription walk gets an output plane, and with it a photometric normalisation (K13, P1)

`propagate_hfpi_through_prescription` bins the path bundle wherever the
surface list leaves it. When that last surface is a diffractor the paths
have just been re-emitted, so the final leg has length zero — and the
Huygens–Fresnel binning Jacobian `r/(dx_out^2 cos theta_out)` that makes
HFPI's amplitudes the HF integral's (K13, applied by the free-space entry
points since 5.46.0) has no `r` to use. The walk could not return a
photometric amplitude at all, and said so in a `RuntimeWarning`.

`z_output` (new, `float | None`, default `None` = today's behaviour)
closes the walk with a `propagate_to_plane` hop to that plane
(`hfpi.py:1941–1954`). The hop is taken in the medium the prescription
puts after its last surface (`glass_after`, so an immersed image space is
handled; a `'MIRROR'` marker resolves to the surface's own
`glass_before`, which is what reflection does to the medium, and folds
the direction instead). `normalisation` gains `'auto'` and becomes the
default (`hfpi.py:1572`): it resolves to `'physical'` when the walk has
both an output plane to close on AND legs that are free space, and to
`'legacy'` — with the existing warning — otherwise.

Measured against band-limited ASM on the same geometry (32×32 at
dx = 4 µm, w0 = 12 µm, z = 2 mm, 0.05 rad cone, unbiased least-squares
complex scale, six seeds at 2 M paths):

| what | scale vs ASM |
|---|---|
| **closed walk**, `dx_out = dx` | **0.9851 … 0.9970** (mean 0.9906) |
| **closed walk**, `dx_out = 2 dx` | **0.9559 … 0.9689** (mean 0.9637) |
| **closed walk**, `dx_out = 4 dx` | **0.8634 … 0.8768** (mean 0.8695) |
| same walk, no `z_output` (5.46.0's only option) | 1.16e-7 |

— flat in output pixel area over a 4× range, which is the K13 property
itself, and flat in path count (0.9841 / 0.9906 / 0.9899 at 0.5 M / 2 M /
8 M, seed-averaged). The oracle-free transparency property holds too: an
*open* stop must be invisible, and the two-leg walk reads 1.0428 ± 0.0499
against the one-leg walk's 0.9908 ± 0.0041 over the same total distance
(ratio 1.0524).

**The condition on the legs is not decoration.** That Jacobian is the
free-space ray-tube relation `dS = r^2 dOmega / cos(theta)`. Put an
element with power between the emission and the landing point and the
system's own Jacobian replaces it, while the per-path factor does not
know. Measured through a 19.41 mm N-BK7 thin singlet (object 60 mm,
2 M paths) against ASM + thin-lens phase + ASM:
`normalisation='physical'` returns **4879×** the reference power at the
image plane and **13.9×** at half that distance, while the spot metrics
stay in the right ballpark (r50 15.9 µm against 17.9 µm, r84 25.3 µm
against 31.9 µm). So `'auto'` refuses to pick `'physical'` there
(`_walk_legs_are_free_space`, `hfpi.py:1151`), and forcing it warns with
those numbers — shape is usable, absolute scale is not.

Cost: the closing hop is one `propagate_to_plane`, measured 1.19–1.32× a
walk without it (80.6 ms against 60.9 ms at 131 072 paths, 806 ms against
679 ms at 1 048 576; medians of five interleaved runs).

Files: `lumenairy/propagators/hfpi.py`.
Tests: `tests/unit/test_audit2609_b3_propagator_kernels.py::TestK13PrescriptionWalkOutputPlane`
(10 tests).

**Migration.** No signature break: every existing call omits `z_output`,
so it still bins at the last surface, still defaults to the legacy sum,
still warns, and returns the same bytes. A call that adds `z_output` gets
a different plane AND — on a flat-optics prescription — photometric
amplitudes, i.e. a scale change of orders of magnitude on purpose; pass
`normalisation='legacy'` to keep the raw path sum. `normalisation`'s
default spelling changed from `'legacy'` to `'auto'`, which resolves to
`'legacy'` for every call that does not pass `z_output`.

---

### Added -- propagators/hfpi: `sampler='sobol'` for the stratified HFPI source draw (K22, P2)

`init_paths_stratified` gains `sampler` (`{'jittered', 'sobol'}`, default
`'jittered'` = unchanged; `hfpi.py:1430`), threaded through
`propagate_hfpi_through_prescription` (`hfpi.py:1569`). `'sobol'` places
`n_paths` scrambled Sobol points in the same 4-D
`(pixel_x, pixel_y, cos theta, phi)` cube the jittered sampler
stratifies (`_sobol_cube_draw`, `hfpi.py:1257`), with the Owen scramble
seeded from `rng` so the bundle stays a pure function of it. `n_paths` is
honoured **exactly** — a low-discrepancy sequence has no stratification
grid to round the count onto, so 1000 paths means 1000 paths where the
jittered 4th-root rule gives 1296.

**The QMC gain was measured before being advertised, and it is not the
textbook one.** Against band-limited ASM on the same 32×32 / 2 mm
geometry, six seeds per point, error as the RMS scatter of the
individual estimates about their own seed-mean over the significant
pixels — the estimator's own error, which is what a convergence rate is
a statement about (the seed-mean's distance from ASM fits the same
exponent to within 0.02):

| n_paths | jittered (actual) | jittered err | sobol (actual) | sobol err |
|---|---|---|---|---|
| 2^14 | 14 641 | 3.978 | 16 384 | 3.774 |
| 2^16 | 65 536 | 1.853 | 65 536 | 1.829 |
| 2^18 | 262 144 | 0.9228 | 262 144 | 0.8546 |
| 2^20 | 1 048 576 | 0.4120 | 1 048 576 | 0.3970 |

Fitted over the last five points, `err ~ N^-p` gives **p = 0.537
(jittered) and p = 0.557 (sobol)** — both Monte-Carlo, neither the
`O(N^-1)` a smooth integrand would give. The error ratio between the two
samplers at matched path count is **≈1.0–1.1×** in Sobol's favour and
fixture-dependent (1.00–1.13× on this fixture, 1.00–1.06× on VERIFY-B3's second one). The
16× error ratio, measured over five independent three-seed groups, is
4.349–4.533 (jittered) and 4.628–4.719 (sobol) against the 4.0 that
`O(N^-1/2)` predicts and the 16.0 that `O(N^-1)` would. The integrand has
hard edges — the cone cut, the output-pixel bin, any aperture — and that
is what the measurement says.

So: **use `'sobol'` for the exact path count, not for a convergence
rate.** Cost is not a reason against it: the draw is 0.80–0.94× the
jittered one (157 ms against 196 ms at 2^20 paths) and a whole
init + propagate + bin estimate is 0.94–1.25×.

Files: `lumenairy/propagators/hfpi.py`.
Tests: `tests/unit/test_audit2609_b3_propagator_kernels.py::TestK22SobolSampler`
(7 tests, including the rate claim as a two-sided bar on the 16× error
ratio).

**Migration.** None — the default is the jittered sampler and its draws
are byte-identical (the existing block moved verbatim into
`_jittered_cube_draw`, `hfpi.py:1312`). `sampler='sobol'` with an
explicit `n_strata_xy` / `n_strata_dir` raises rather than ignoring them,
and a non-power-of-two `n_paths` warns, because a Sobol sequence is
balanced only on its `2**m` prefixes.

---

### Added -- propagators/rs: `kernel='spatial-integrated'`, the Shen--Wang pixel-integrated RS kernel (K9 second half, P2)

`rayleigh_sommerfeld_propagate` gains a fourth `kernel` token
(`rs.py:679`) that integrates the RS-I Green's function over each pixel
instead of sampling it at the centre (reference [2], Shen & Wang 2006),
by a folded 6-node tensor Gauss-Legendre rule
(`_rs_pixel_integrated_kernel`, `rs.py:268`) built on one quadrant and
mirrored, with its own `'RS_INT'` H-cache tag (`rs.py:812`). The alias
guard now names whichever spatial kernel was asked for and covers both
(`rs.py:783`), since the pixel integral narrows the kernel's spectrum by
a sinc without band-limiting it.

**It is not a more accurate `'spatial'`; it answers a different
question,** and which one is right is a property of the caller's array.
`'spatial'` reads `E_in` as point samples of a smooth field, so the sum
is a trapezoidal rule and is spectrally accurate.
`'spatial-integrated'` reads `E_in` as cell values of a field that is
constant across each pixel — a binary mask, a pixelated DOE or SLM map —
and the convolution is then that field's exact RS integral. Measured
(λ = 633 nm, circular aperture a = 100 µm, window 512 µm, z = 16 mm,
which is above the alias threshold of every grid quoted):

*Against a super-sampled continuum RS-I double quadrature of the
STAIRCASE aperture (a midpoint rule with `S` sub-samples per pixel axis,
direct summation, no FFT; its own floor read off from `S` against `2S`):*

| grid | S | oracle floor | `'spatial-integrated'` | `'spatial'` |
|---|---|---|---|---|
| N = 64, dx = 8 µm | 16 | 1.3794e-5 | **4.5979e-6** | 4.7464e-3 |
| N = 64, dx = 8 µm | 32 | 3.4485e-6 | **1.1495e-6** | 4.7499e-3 |
| N = 128, dx = 4 µm | 16 | 3.3558e-6 | **1.1186e-6** | 1.1465e-3 |
| N = 128, dx = 4 µm | 32 | 8.3895e-7 | **2.7965e-7** | 1.1473e-3 |

The integrated kernel sits *below* the oracle's own floor and divides by
four every time `S` doubles — that is the oracle converging onto it. The
point-sampled kernel does not move with `S` at all and stands 1024× /
4103× away.

*Against an exact Hankel angular-spectrum quadrature of a Gaussian, the
ranking reverses by four to five decades:* `'spatial'` reads 3.2e-8 …
6.6e-8 across six legal geometries where `'spatial-integrated'` reads
8.2e-4 … 1.3e-2, because it convolves the staircase of the Gaussian
rather than the Gaussian.

**`'auto'` therefore does NOT select it and the default does not move.**

**And it is not what fixes the audit's hard-aperture convergence.** That
was measured as roughly first order (2.86e-2 → 1.08e-3 over
N = 256…2048); the cause is the APERTURE's edge, not the kernel. On-axis
relative error against the closed form
`U = e^{ikz} - (z/r_a) e^{ik r_a}`, same fixture:

| N | dx [µm] | stair + point | stair + integrated | grey + point | grey + integrated |
|---|---|---|---|---|---|
| 128 | 4.000 | 8.3008e-3 | 8.4184e-3 | 1.4045e-3 | 2.6828e-3 |
| 256 | 2.000 | 3.3548e-3 | 3.3707e-3 | 3.4263e-4 | 6.6308e-4 |
| 512 | 1.000 | 3.4207e-4 | 3.5238e-4 | 8.4251e-5 | 1.6429e-4 |
| 1024 | 0.500 | 5.2718e-4 | 5.2758e-4 | 2.0677e-5 | 4.0724e-5 |

Measured order between successive rows: **1.307 / 3.294 / −0.624** and
1.321 / 3.258 / −0.582 for a pixel-centre-indicator aperture (erratic and
non-monotone, because a circle's staircase area error does not shrink
smoothly) against **2.035 / 2.024 / 2.027** and **2.016 / 2.013 / 2.012**
for its exact pixel-AREA average. The lever that restores second order is
the input's edge — 25× at N = 1024 — for *either* kernel, and the kernel
choice then moves the constant by ~2×, in the point-sampled kernel's
favour. The library already builds that input:
`apply_aperture(..., edge='gray')` gives each rim pixel its supersampled
open-area fraction, and both docstrings now say to reach for it before
reaching for this kernel. The same measurement on `hf.py`'s OPL
quadrature reads 2.7708e-2
→ 1.7601e-3 (orders 1.32 / 3.27 / −0.61) for the indicator against
1.4762e-2 → 2.2423e-4 (2.03 / 2.02 / 1.99) for the area average; that
function's docstring now carries it.

Cost: kernel build 18.5 / 146 / 782 ms against 3.4 / 16.7 / 72.0 ms for
the point sample at N = 128 / 256 / 512 (5.5× / 8.7× / 10.9×, medians of
five interleaved runs). End to end that is 2.4× and 4.2× a whole
`'spatial'` call at N = 128 / 256 on a cold H cache, and 0.65× / 0.93× —
the same call — on a warm one.

`kernel=` already reaches the RS kernel through
`propagate_huygens_fresnel_freespace`'s `**kwargs`, so the new token is
available there with no wrapper; that is now stated in its docstring.

Files: `lumenairy/propagators/rs.py`, `lumenairy/propagators/hf.py`
(docstrings only — both fingerprints unchanged).
Tests: `tests/unit/test_audit2609_b3_propagator_kernels.py::TestK9PixelIntegratedRsKernel`
(9 tests).

**Migration.** None. `'auto'`, `'spatial'` and `'transfer'` are
byte-identical, including at odd `N`, anamorphic pitch, `complex64` and
`bandlimit=True`. `'spatial-integrated'` is opt-in, refuses inside the
same alias regime `'spatial'` does, and should be reached for only when
the input array's staircase IS the object.

---

### Added -- propagators/mft: `resample_field(method='chirpz')`, a band-limited resampler (K6 second half, P2)

`resample_field` interpolates the real and imaginary parts separately
with a cubic spline, so a near-Nyquist complex carrier is attenuated —
K6 measured 0.9 % to 6.8 % of the power lost at the "≥ 4 px per feature"
its docstring rated at "< 0.1 %", and the single-FFT Fresnel output
chirp sits at exactly Nyquist at the grid edge by construction.

`method` (new, `{'spline', 'chirpz'}`, keyword-only, default `'spline'`;
`mft.py:550`) adds the band-limited alternative: transform to the centred
spectrum and inverse-transform it straight onto the output grid with
`_bluestein_centred_2d` (`_resample_field_chirpz`, `mft.py:489`), which
is the trigonometric (Dirichlet-kernel) interpolant of the samples. Its
MTF is exactly 1 at every frequency the input grid represents. Measured
on the same Gaussian-times-carrier fixture K6 used (power ratio after
resampling):

| carrier (cyc/px) | px per cycle | `'spline'` | `'chirpz'` |
|---|---|---|---|
| 0.00 | ∞ | 0.999999 | **1.000000** |
| 0.10 | 10 | 0.999497 | **1.000000** |
| 0.20 | 5 | 0.990258 | **1.000000** |
| 0.30 | 3.33 | 0.930135 | **1.000000** |
| 0.40 | 2.5 | 0.717718 | **1.000000** |

— and the same story down-sampling at scale 1.5, where the spline reads
0.999999 / 0.999440 / 0.989332 / 0.924956 / 0.700854 on those five rows
against a flat 1.000000. Against the explicit Dirichlet-kernel double
sum the
chirp-Z leg agrees to ~1e-14 relative, and `dx_out == dx_in` returns the
input to 9.5e-15.

Two properties to know before switching a call site. The chirp-Z
reconstruction is **periodic** with period `N_in*dx_in`, so an output
window wider than the input extent returns replicas rather than the
zeros the spline pads with — measured power ratio exactly 4.000000 for a
2× window, where the spline gives 1.000000. That case now warns, reusing
the MFT family's faithful-zone diagnostic (`mft.py:769`), which grew a
per-axis `N_out_y` for the non-square extent-preserving default
(`mft.py:95`). And neither leg anti-aliases on down-sampling.

Cost: 1.07× at N = 64 and N = 128, 1.20–1.47× at N = 256–512 upsampling,
1.78× down-sampling at N = 512 (medians of seven interleaved runs).

Files: `lumenairy/propagators/mft.py`.
Tests: `tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler`
(7 tests, the default leg pinned by BYTE identity against
`map_coordinates` driven directly).

**Migration.** None — `method='spline'` is the default and is
byte-identical, as are all three MFT propagators past the shared warning
helper. The `fresnel` / `sas` resample-back legs in
`propagators/system.py` and `elements/_lens_real.py` are **not**
switched by this release; see WP-B3's report §5 for the measurement that
says the switch has to be conditional on the direction of the pitch
change, and for a stronger alternative on the Fresnel leg.

---

### Changed -- CONVENTIONS / docs

Nothing in `CONVENTIONS.md` moves. Three history documents are
re-recorded in this change (`docs/history/lumenairy.propagators.rs.md`,
`.mft.md`, `.hfpi.md` — the last with two `re_recorded:` lines).
`lumenairy.propagators.hf.md` is not: the recorder was run on it and
answered `OK`, because both fingerprints strip docstrings and the hf.py
edits are docstrings only. `record_history_fingerprints.py --check`
lists all four `OK`.

One cross-module documentation point worth surfacing at release: the K9
measurement says the largest accuracy lever available to a caller of
`rayleigh_sommerfeld_propagate(kernel='spatial'|'spatial-integrated')`
or of `propagate_huygens_fresnel_with_opl_callable` is the aperture's
edge, and `apply_aperture(edge='gray')` already builds it. Whether
`edge` should DEFAULT to `'gray'` is a question for
`elements/elements.py`'s owner, not this release.

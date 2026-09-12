# ANALYSIS audit — `lumenairy/analysis/` (17,839 lines)

*(The auditor's own file write was blocked by the harness; this file reproduces the report returned by the auditor verbatim.)*

Repro scripts: `…/scratchpad/ANALYSIS/p1_zernike.py` … `p12_objdist.py`. All numbers measured on this box (CPython 3.14, numpy 2.4.6, scipy 1.17.1, CPU only).

## Scope read

**Complete, line by line:** `zernike.py` (1–859), `opd.py` (1–701), `strehl.py` (1–543), `beam_stats.py` (1–729), `psf_mtf_otf.py` (1–1579), `through_focus.py` (1–1883), `aberration.py` (1–667), `polychromatic.py` (1–667), `detector.py` (1–803), `ao.py` (1–1355), `interferometry.py` (1–234), `coherence.py` (1–223), `coronagraph.py` (1–198), `core.py` (1–69), `field.py` (1–1492), `ghost.py` (1–949), `image_plane_wfe.py` (1–1042), `__init__.py` (1–250).

`phase_retrieval.py`: 1–860 line by line; 860–1021 (JAX argument plumbing) skimmed. `plotting.py`: skimmed per instructions — read 1–270, 555–760 plus a grep pass over every `imshow`/`extent`/`origin` site; no plotting function executed.

**Not reached:** JAX code paths desk-checked only; `plotting.py` rendering beyond the extent/orientation sweep.

---

## Findings

### [P0] `wave_opd_2d` row-then-column unwrap produces per-column 2π slips on ANY aberrated pupil — `lumenairy/analysis/opd.py:689-690`

```python
phase_unwrapped = np.unwrap(phase, axis=1)
phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)
```

**What is wrong.** `phase = np.angle(field)` is taken over the *whole grid*, including outside the pupil where `|E| = 0` and `np.angle(0) = 0`. The row pass is anchored at column 0 (outside the pupil), so each in-pupil row picks up its own 2π·kᵢ offset. The column pass is then **anchored independently in every column**, at that column's first in-pupil row, and the wrap of the entry jump injects a *column-dependent* 2π multiple. Masking to `valid` happens only afterwards.

Row-then-column is valid on a *fully supported* field; the zero-amplitude exterior is what breaks it (1 wave rms coma, N=512):

| pupil | max \|OPD err\| |
|---|---|
| amplitude 1 everywhere, `aperture=None` | 3.7e-14 waves |
| amplitude 1 everywhere, `aperture=` crop after unwrap | 1.1e-14 waves |
| **normal pupil (zero exterior)** | **4.0 waves** |

**Evidence** (`p2c_unwrap_threshold.py`, `p2b_unwrap.py`). Flat circular pupil, N=512, dx=1 µm, aperture 400 µm, λ=633 nm, **no defocus**, only primary coma (OSA j=8). Per-sample phase gradient is 10–25× *below* the π Nyquist limit and `check_opd_sampling` reports SAFE at every row:

| coma [λ rms] | max \|dφ\|/sample | max \|OPD err\| | % pixels wrong >0.4 λ | fitted c8 (true) |
|---|---|---|---|---|
| 0.10 | 0.062 rad | 0.000 λ | 0.00 % | +0.1000 (0.1000) |
| **0.20** | **0.124 rad** | **1.000 λ** | **5.2 %** | **+0.1024 (0.2000)** |
| 0.30 | 0.185 rad | 1.000 λ | 29.1 % | +0.1646 (0.3000) |
| 0.50 | 0.309 rad | 1.000 λ | 55.2 % | +0.5489 (0.5000) |
| 1.00 | 0.618 rad | 4.000 λ | 76.6 % | +0.9979 (1.0000) |
| 5.00 | 3.090 rad | 19.000 λ | 95.2 % | +4.8492 (5.0000) |

Errors are **exactly integer waves** and **constant within each column** (`p2b`: "columns with a single constant wave-offset: 641/641; rows: 210/641") — the fingerprint of the per-column anchor.

The `f_ref=` path used by the optimiser does **not** save it:

| residual coma [λ rms] | max \|err\| | % wrong | fitted RMS-WFE (true) |
|---|---|---|---|
| 0.20 | 1 λ | 4.4 % | **0.485 λ** (0.200) |
| 0.50 | 1 λ | 55.1 % | **0.988 λ** (0.500) |
| 1.00 | 3 λ | 76.9 % | **1.752 λ** (1.000) |
| 2.00 | 7 λ | 88.9 % | **3.447 λ** (2.000) |

**Impact.** Only 2-D OPD extractor in the library. Consumers: `optimize/driver.py:945` (→ `ctx.opd_map` → `RMSWavefrontMerit`, `ZernikeCoefficientMerit`, `MatchTargetOPDMerit`), `optimize/wrapper_merits.py:530, 802, 1008`, GUI `wavefront_map_dock.py:461` and `field_browser_dock.py:181` (both **without** `f_ref`), `interferometry_dock.py:159`, and the `docs/cookbook.md:147` recipe. An optimiser driven by a 2× inflated RMS-WFE converges to the wrong prescription.

**Prior-audit status.** `docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md:282` logs "S2.4: … path-dependent for branch-cut wavefronts (documented limitation)". That is **wrong**: the failing wavefronts are smooth, single-valued, simply connected, no branch cut. The docstring at `opd.py:633-635` ("for diagnostic OPD maps over small apertures a simple row-then-column unwrap is adequate") is false — it fails at 0.2 waves rms over a 400 µm aperture.

**Fix** (measured exact): anchor every row on one unwrapped column instead of a second full `unwrap(axis=0)`:
```python
jc = Nx // 2
col = np.unwrap(phase[:, jc])
out = np.unwrap(phase, axis=1)
out = out + (col - out[:, jc])[:, None]
```
→ **max error 1.0e-15 waves** on the case the current code gets wrong by 4 waves. One extra 1-D unwrap plus a broadcast add. Better still: mask before unwrapping (`np.ma` + `skimage.restoration.unwrap_phase`) or a least-squares unwrap for annular/multiply-connected supports.

---

### [P0] `diffraction_limited_peak` uses a PARAXIAL quadratic reference — every ASM-based Strehl is inflated, by 10× at f/5 — `lumenairy/analysis/through_focus.py:272`

```python
E_ideal = np.abs(E) * np.exp(-1j * k0 * (X ** 2 + Y ** 2) / (2.0 * f))
```

**What is wrong.** The aberration-free reference is the *spherical* wave `exp(-i k0 (√(r²+f²) − f))`. The quadratic approximation differs by a genuine spherical-aberration term `W040 = (D/2)⁴/(8 f³ λ) = (D/λ)/(128 (f/#)³)` waves, so the "diffraction-limited" denominator is **itself aberrated**, its peak is depressed, and Strehl is inflated by `1/S_ref`. The propagation (`angular_spectrum_propagate`) is exact/non-paraxial, so nothing cancels it.

**Evidence — end-to-end, default path** (`p6_strehl_gt1.py`). A **perfect, aberration-free** exact-spherical converging wavefront (D = 614 µm, f = 1.228 mm, f/2.0, λ = 600 nm, N = 4096, dx = 0.2 µm, W040 = 0.999 waves):

```
diffraction_limited_peak (paraxial ref)           = 1.342173e+04
peak of the PERFECT sphere at z = f               = 1.519335e+05
through_focus_scan best Strehl for a PERFECT lens = 11.3200   at z = 1228.00 um
```

**Strehl 11.32 for a diffraction-limited pupil.** Correct answer: 1.0000.

**Evidence — magnitude vs geometry** (`p5_detector_ref.py`). `S_ref` from `strehl_phase_integral` on the paraxial-minus-sphere phase error, cross-validated against a direct ASM propagation (0.98665 measured vs 0.98671 predicted; 0.99982 vs 0.99979 on two resolvable geometries):

| D | f | f/# | W040 [λ] | S_ref | Strehl inflated by |
|---|---|---|---|---|---|
| 10 mm | 200 mm | f/20 | 0.016 | 0.99907 | 1.001× |
| 10 mm | 100 mm | f/10 | 0.130 | 0.94221 | **1.06×** |
| 10 mm | 50 mm | f/5 | 1.042 | 0.09482 | **10.5×** |
| 25.4 mm | 100 mm | f/3.9 | 5.420 | 0.02784 | **35.9×** |
| 10 mm | 25 mm | f/2.5 | 8.333 | 0.01572 | **63.6×** |

(λ = 600 nm. A 1-inch f/4 optic — the ordinary case for this repo's free-space work — sits in the 36× row.)

**Impact.** Strehl denominator for `through_focus_scan`/`_jax`/`single_plane_metrics`, `tolerancing_sweep` (`:835, 866`), `monte_carlo_tolerancing` (`:992`), `_jax` (`:1456`), `_linearized` (`:1738`), `polychromatic_strehl` (`polychromatic.py:208`), `polychromatic_psf`'s `per_wavelength_strehl` (`:392`), and the optimiser's `StrehlMerit` via `ctx.strehl_best`.

**Prior-audit status.** The symptom has been seen and *mis-diagnosed* repeatedly. `CHANGELOG.md:14317` blamed a per-trial denominator and a hard-coded focal length; `CHANGELOG.md:14147` declared the residual "NOT a normalization bug… a methodology consequence"; `through_focus.py:609-611` says "Strehl can exceed 1 in some edge cases (e.g. the ideal reference is on-axis but the aberrated peak happens to concentrate)". None is the mechanism, and none bounds the error — it is set purely by `(D/λ)/(128 (f/#)³)`.

**Fix** (one line):
```python
E_ideal = np.abs(E) * np.exp(-1j * k0 * (np.sqrt(X**2 + Y**2 + f*f) - abs(f)))
```
Exact at every f/#, one `sqrt` per pixel, reduces to the present form in the paraxial limit. Add a regression test asserting `Strehl == 1.0 ± 1e-3` for an exact-sphere pupil at f/2.

---

### [P1] `shack_hartmann`'s reconstructed `wavefront` is a factor of 2 too small — `lumenairy/analysis/detector.py:796-801`

```python
wf_x = np.cumsum(sx_safe, axis=1) * pitch_actual
wf_y = np.cumsum(sy_safe, axis=0) * pitch_actual
wf_x -= wf_x[0, 0];  wf_y -= wf_y[0, 0]
wavefront = 0.5 * (wf_x + wf_y)
```

`wf_x[i,j] ≈ W(x_j,y_i) − W(x_0,y_i)` and `wf_y[i,j] ≈ W(x_j,y_i) − W(x_j,y_0)`. For any **separable** wavefront `W = f(x)+g(y)` — tilt, defocus, astigmatism, i.e. essentially every use — their average is `½[f_j+g_i−f_0−g_0] = ½(W − const)`, **exactly half**. Anchoring removes each half's piston but not the ½.

**Evidence** (`p4_mixed.py` §A). N=256, dx=5 µm, pitch = 32 dx, f = 5 mm, λ = 632.8 nm; ratio fitted against the slope gain measured on the same run:

| input | slope gain | wavefront/expected | expected if correct |
|---|---|---|---|
| tilt 0.20 mrad | 0.9481 | **0.4741** | 0.948 |
| tilt 0.50 mrad | 0.9488 | **0.4744** | 0.949 |
| tilt 1.00 mrad | 0.9450 | **0.4725** | 0.945 |
| defocus, 1 µm edge | — | **0.4626** | ~0.95 |

`0.5 × gain` in every case; `max|wf| = 5.43e-7 m` where the truth is `1.50e-6 m`.

**Impact.** `slopes_x/slopes_y` (what the AO stack consumes) are correct; only the third return value is wrong. But `wavefront` is a documented primary output and anyone budgeting from it is 2× optimistic. The comment concedes the average "is not a valid 2-D reconstruction" but never says the *scale* is wrong by 2; "cumulative trapezoidal" is also inaccurate (it is a rectangle-rule `cumsum`).

**Fix.** Integrate along one edge then perpendicular (`W[i,j] = cumsum_x(sx)[i,j] + cumsum_y(sy)[:,0][i]`, the correct Itoh path integral, no ½), or do a least-squares Southwell/Fried/Hudgin solve. At minimum drop the `0.5 *` and document the convention.

---

### [P1] `eval_image_plane_wfe` silently returns garbage for `object_distance ≳ 1e4 m` and has no infinite-conjugate mode — `lumenairy/analysis/image_plane_wfe.py:370-374, 503-508`

The function **requires** `object_distance > 0` (raises otherwise) and launches the bundle *at the object* (`bundle.z = np.full(px.size, -obj_d_m)`, `:506`). A user modelling an infinite conjugate has no option but a large finite number — and the ray–sphere intersection then suffers catastrophic cancellation (`c = |P−C|² − R²` with `|P−C|² ~ 1e12`, `R² ~ 2.5e-3`; float64 resolves 1e12 only to 2.2e-4 m).

**Evidence** (`p12_objdist.py`). Biconvex N-BK7 singlet R=±50 mm, d=3 mm, D=10 mm (f/4.8), λ=587.6 nm. Last column is the **on-axis chief**, which must land at z = 0 by definition:

| object_distance | z0 [µm] | analytic sag [µm] | error [µm] | chief z0 [µm] |
|---|---|---|---|---|
| 1e0 | 226.24 | 225.73 | +0.51 | +0.0000 |
| 1e3 | 226.14 | 225.63 | +0.51 | +0.0006 |
| 1e4 | 225.99 | 225.63 | +0.36 | +0.0238 |
| 1e5 | 224.04 | 225.63 | −1.58 | **−5.34** |
| 1e6 | **589.41** | 225.63 | **+363.8** | **+589.41** |
| 1e7 | **50000.0** | 225.63 | **+49774** | **+50000** |

Resulting wavefront error:

| object_distance | img_d [mm] | PV [λ] | RMS [λ] |
|---|---|---|---|
| 1e2 – 1e3 | 47.88–47.90 | 3.47 | 1.043 |
| 1e4 | 47.8754 | 3.69 | 1.065 |
| 1e5 | 47.8752 | **47.27** | **10.76** |
| 1e6 | 47.8752 | **213.51** | **61.91** |

The correct answer (independent transverse-ray-aberration oracle, `ε = −(R/a)·dW/dρ` integrated over the pupil, `p10_wfe_dissect.py`) is **PV = 3.72 waves, W040 = 3.71 waves**, cross-checked against the traced longitudinal SA (−0.87 mm ⇒ W040 = 3.52 waves). At `object_distance = 1e6` the library is **62× high**, and the excess is a pure `ρ²` defocus (verified: library-minus-oracle is exactly quadratic in ρ). Downstream, `image_plane='best_rms'` "corrects" it by moving the reference sphere to **96.6 mm** on a lens whose BFL is 47.9 mm — the returned `img_d_m` is unphysical and the returned RMS (0.91 waves) meaningless. No warning is emitted.

Confirmed **correct in its valid regime** (`object_distance = 1 m`): agrees with the oracle to ~3 % across the pupil, documented sign holds (+1.99 waves at ρ=0.88 for the undercorrected singlet).

**Fix.** Accept `object_distance = inf`/`None` and launch a collimated bundle at the entrance pupil, adding the constant object-side OPL analytically. Failing that, launch a few mm before surface 0 and warn/raise when `object_distance` exceeds ~1e4 × EFL.

---

### [P2] `gerchberg_saxton`'s reported `error`/`history` are off by N² and cannot converge to zero — `phase_retrieval.py:182-187, 211, 232`

```python
target_scaled = target_amplitude * np.sqrt(source_power / target_power)
err = np.mean((np.abs(far_field) - target_scaled) ** 2)
```
`far_field` is an **unnormalised** DFT (`sum|F|² = N²·sum|field|²`), so comparing it against a target rescaled to the *source* power leaves a hard-wired factor N.

**Evidence** (`p4_mixed.py` §E). N = 64, target built as `|FFT(source·e^{iφ0})|` so the supplied `initial_phase = φ0` is an **exact** solution:
```
true |FFT| vs target max abs diff = 2.132e-14      (the solution IS exact)
reported final error              = 3.775380e+02
mean(target^2)                    = 3.896184e+02
history[0], history[-1]           = 3.7754e+02, 3.7754e+02    <-- flat
target_power / source_power       = 4.0960e+03 = N^2
```
The reported error is ~97 % of the target energy and does not move over 50 iterations. **The retrieved phase is unaffected** (both amplitude-replacement steps are scale-invariant), which is why this survived. `error_reduction`/`hybrid_input_output` do not rescale and their metric is correct (history 3.71e2 → 1.70e0). The JAX twin `gerchberg_saxton_jax` (`:789-790`) also does not rescale, so the two backends' `err` differ by N² despite "Same physics as `gerchberg_saxton`".

**Fix.** `target_scaled = target * np.sqrt(N_pix * source_power / target_power)`, or compare `|F|/N`. Either makes `err → 0` for an exact solution and restores backend parity.

---

### [P2] `single_plane_metrics` computes `|E|²` three times and the centroid twice per plane — `through_focus.py:193-202`

```python
I = np.abs(E) ** 2                       # pass 1
cx, cy = beam_centroid(E, dx, dy)        # pass 2 + centroid
d4x, d4y = beam_d4sigma(E, dx, dy, ...)  # pass 3 + centroid again
```
`beam_d4sigma` recomputes the centroid internally (`beam_stats.py:331-332`).

**Measured** (N = 1024 complex128, `p7_perf.py`): `single_plane_metrics` = 158–176 ms/plane, of which `beam_centroid` = 48 ms and `beam_d4sigma` = 92 ms; a single `np.sum(X*I)` is 4.5 ms and `np.abs(E)**2` ~25 ms. A single-pass version is ≈ 50 ms → **~3× faster**, ~14 % off a through-focus scan (785 ms/plane warm). Fix: add an optional `centroid=` to `beam_d4sigma` and share a private `_moments(I, X, Y)` helper.

---

### [P2] `through_focus_scan`'s per-plane transfer function is 70 % of the propagation cost and is rebuildable by recurrence — `through_focus.py:451-457`

**Measured** (N = 1024, isolated loop body):

| step | ms |
|---|---|
| **`np.where(prop, np.exp(1j·kz·z), 0)`** | **185.7** |
| bandlimit mask multiply | 19.4 |
| `E_fft * H` | 11.5 |
| `ifftshift` | 10.6 |
| `_ifft2` | 46.8 |
| `fftshift` | 17.1 |
| **full body** | **264.5** |

The `exp` alone costs 4× the IFFT. Since `z_values` is almost always a `linspace`, `H(z_{n+1}) = H(z_n)·H(Δz)` — **one complex multiply, measured 11.2 ms** — replaces it: ~175 ms/plane, **~22 % of the warm scan**. Even without the recurrence, `np.exp` + `np.where` allocate two full complex grids per plane; `np.exp(..., out=buf)` plus in-place masking removes both. `bl_x[None,:] & bl_y[:,None]` also materialises a full N² bool per plane and can be two 1-D in-place multiplies. Recurrence error over 21 planes: ~21·2.2e-16 — nothing in float64. Guard behind "z uniformly spaced".

---

### [P2] `encircled_energy_curve` / `encircled_energy_radius` each rebuild a full `argsort` of N²; curve+radius sorts twice — `psf_mtf_otf.py:427-429`

| N | curve | radius |
|---|---|---|
| 512 | 59 ms | 54 ms |
| 1024 | 337 ms | 216 ms |
| 2048 | 1567 ms | 1394 ms |

The docstrings advertise `_ee_sorted_cumulative` as "the single source of truth" shared by both — it is *re-executed*, not cached. Related: `radial_power_bands` (`polychromatic.py:109-111`) builds a **full N² boolean mask per radius** in a Python loop — 65.9 ms for 1 radius, 151 ms for 64 at N=1024; the sort-and-`searchsorted` construction would make it O(N² log N) once, independent of `n_radii`.

---

### [P2] `compute_psf` materialises ~4 full padded complex grids — `psf_mtf_otf.py:161-172`

| oversample | N_psf | time | peak traced alloc |
|---|---|---|---|
| 1 | 1024 | 176 ms | 50 MB |
| 2 | 2048 | 1254 ms | 268 MB |
| 4 | 4096 | 3978 ms | **1074 MB** |

One 4096² complex128 grid is 268 MB → peak is 4 grids (`pad`, `ifftshift`, `fft2` out, `fftshift`). The shift pair is removable: `fftshift(fft2(ifftshift(a))) ≡ chess*fft2(chess*a)` with `chess = (−1)^(i+j)`, applicable in place, saving ~536 MB at oversample 4. At N = 32768 one complex128 grid is 17.2 GB.

---

### [P2] `plot_stokes` uses the x-axis extent for both axes — still unfixed since AUDIT_V4_13_0 — `plotting.py:718, 733, 738`

```python
extent, unit_label, _ = _auto_extent(Nx, dx, unit)   # Nx and dx only
im = ax.imshow(data, extent=extent, origin='lower', ...)
```
`AUDIT_V4_16_0_DEEP_2026_05_19.md:279` lists this as "**S2.1 (real bug)**: `plot_stokes` extent uses `Nx` for both axes; wrong for `Ny≠Nx` Jones fields". Still present, and there is no `dy` parameter, so a non-square *or* anamorphic Jones field gets a mislabelled y axis. Verified: `_auto_extent(64, 1e-6) == _auto_extent(32, 2e-6) == (-32,32,-32,32) µm` — indistinguishable. `plot_intensity`/`plot_phase` (`:172-174`, `:246-248`) already do the right thing with a second `_auto_extent(Ny, dy, …)`.

---

### [P3] Two incompatible `imshow` extent conventions in `plotting.py`, both off by half a pixel — `plotting.py:113-114` vs `:602-603`
`_auto_extent` returns `(−N/2·dx, +N/2·dx)` while the sample grid is `(arange(N) − N/2)·dx`. Since `extent` addresses *outer edges*, drawn pixel centres are `(−N/2+0.5)dx … (N/2−0.5)dx` — every plot shifted by **+dx/2** (verified: samples `[-4…3]`, drawn centres `[-3.5…3.5]`). `plot_psf` uses `extent = (x[0], x[-1])` — the opposite half-pixel error. Correct: `(x[0] − d/2, x[-1] + d/2)`. `detector.py:124-132` notes this anchor class was deferred by AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1; the remaining defect is that the two sites disagree, so the same field plotted two ways lands one pixel apart.

### [P3] `ghost.retrace_ghost_path` reports a "50 % encircled-energy" FWHM from the median **ray** radius — `ghost.py:934-937`
`r_half = np.median(radii); fwhm = 2*r_half`. `make_rings` puts the same `rays_per_ring` on every ring, so ray density per unit *area* falls as 1/r; the median ray radius is not the 50 %-EE radius (under-reads ~17 % on a uniform disc). Weight rays by ring area first.

### [P3] `aberration.caustic_diagnostic` silently clamps a complex-eigenvalue Jacobian — `aberration.py:544`
`disc = max(0.25*tr*tr - det, 0.0)` collapses complex eigenvalues to `tr/2`, so a skewed system reports spurious coincident eigenvalues and a wrong Maslov index. Benign for axisymmetric systems; should flag `disc < 0`.

### [P3] `simulate_interferogram` has no `dy`, and its documented output range is wrong — `interferometry.py:66, 58-59`
`y = (np.arange(Ny) - Ny/2) * dx` uses `dx` for the y tilt. Docstring promises "values in [0, 1]"; the implementation returns `background·(1 + V·cos φ)` ∈ `[0, 2·background]`.

### [P3] Three public `analysis` functions reachable only from the package root
`ao_closed_loop`, `make_shack_hartmann_wfs`, `coronagraph_contrast_curve` are exported by `lumenairy/__init__.py` but **absent** from `lumenairy/analysis/__init__.py` (verified at runtime), though they live in `analysis/`. Likewise `clear_meshgrid_cache` / `meshgrid_cache_bytes` / `zernike_basis_cache_bytes` are exported nowhere.

### [P3] `strehl_phase_integral` vs `strehl_ratio` disagree by 3 orders of magnitude on a tilted wavefront, undocumented — `strehl.py:158-209`
1 wave rms of pure tilt: peak-ratio **0.99996**, `strehl_phase_integral` **0.00069**. Both conventions are legitimate, but the docstring never warns that piston/tilt are *not* removed.

### [P3] `zernike_basis_matrix`'s cache-key "mid-point sample" carries no information for X — `zernike.py:240-251`
`_mid = Xa.size // 2` indexes `X.flat[N·N/2] = X[N/2,0] = x[0]` — identical to `X.flat[0]` for a row-repeating meshgrid. Use a non-repeating index or `float(Xa.sum())`.

---

## Performance opportunities

| # | Site | Measured now | After | Basis |
|---|---|---|---|---|
| 1 | `through_focus.py:451` `H_z` rebuild | 185.7 ms/plane (N=1024) | 11.2 ms/plane via `H *= H_step` recurrence | direct timing of both |
| 2 | `through_focus.py:193-202` metrics | 158 ms/plane | ~50 ms (single pass) | `sum(X*I)` = 4.5 ms ×5 + 25 ms `abs()**2` |
| 3 | `psf_mtf_otf.py:171` shift pair | 1074 MB peak at N_psf=4096 | ~540 MB via in-place chessboard sign | `tracemalloc` |
| 4 | `psf_mtf_otf.py:427` EE sort | 1.57 s + 1.39 s at N=2048 | one sort with a small LRU | direct timing |
| 5 | `polychromatic.py:109` `radial_power_bands` | 66 ms (1) → 151 ms (64) at N=1024 | ~70 ms for any count | direct timing |
| 6 | `zernike.py:88-95` `_zernike_radial` | basis build 1.81 s (21 modes, 502 655 px) | several× via Kintner / Prata–Rusch recurrence | direct timing |
| 7 | `ao.py:167-185` DM IF cache | 537 MB eager for 16×16 DM on 512² (at the 512 MB ceiling) | banded build (exists in `fit_phase`) | `tracemalloc` |

#1 + #2 removes ~30 % of a NumPy through-focus scan at N = 1024 with no accuracy change. The v4.12.2 FFT/K-grid hoist is correct, and `E_fft_shifted` is properly `.copy()`'d off the pyFFTW plan buffer (`:436`).

---

## Alternative algorithms / methods

1. **2-D phase unwrapping** (fixes P0 #1). Cheapest exact fix: Itoh path integral anchored on one column (1e-15 waves). Robust: Goldstein–Zebker–Werner branch-cut (1988) or quality-guided Herráez, Burton, Lalor & Gdeisat, *Appl. Opt.* 41 (2002) 7437 — `skimage.restoration.unwrap_phase`, which accepts a **mask**, directly solving the annular/obscured-pupil case. Least-squares: Ghiglia & Romero, *JOSA A* 11 (1994) 107 — DCT-based, O(N² log N), residue-immune.
2. **MFT instead of zero-padded FFT for the PSF.** `compute_psf` does **not** use the library's own `propagators/mft.py` (`fraunhofer_propagate_mft` exists; verified `'mft' not in inspect.getsource(compute_psf)`). Soummer, Pueyo, Sivaramakrishnan & Vanderbei, *Opt. Express* 15 (2007) 15935: arbitrary focal-plane pitch/extent, O(N²·M), no padding — 1074 MB → ~0 extra at oversample 4, and it removes the power-of-2 pressure. Also the standard tool for the coronagraph pipeline `coronagraph.py` already assumes ("MFT-zoomed focal grids", `:139-140`).
3. **Zernike by recurrence.** Kintner, *Opt. Acta* 23 (1976) 679, or Prata & Rusch, *Appl. Opt.* 28 (1989) 749 — stable to n ≈ 100 and faster than the factorial sum. (Current code is *accurate* at sizes tested — Gram max off-diagonal 1.5e-3 at N=512, pure discretisation.)
4. **Strehl reference.** Beyond the sphere fix, the Debye–Wolf / Richards–Wolf reference (`richards_wolf_focus` already exists) is correct for NA ≳ 0.6.
5. **Shack–Hartmann reconstruction.** Southwell, *JOSA* 70 (1980) 998 (or Fried 1977 / Hudgin 1977) least-squares geometry — one sparse solve, correct in scale, valid for non-separable wavefronts.
6. **Encircled energy**: the sorted-cumsum construction is already optimal; only caching remains.

---

## Code organization observations

- **`through_focus.py` (1883 lines) mixes three concerns** — single-plane metrics, the z-scan, and ~1100 lines of tolerancing (deterministic, MC, JAX MC, linearised MC, report formatting). The tolerancing half imports `apply_real_lens`, which is what gives `analysis` a hard dependency on `elements`. An `analysis/tolerancing.py` split would mirror the v5.1 `core.py` split.
- **Comment-to-code ratio.** `detector.py:389-538` is a 150-line docstring for a 120-line function, mostly a changelog of two fixed bugs. `psf_mtf_otf.py:963-982` is a 20-line comment in a 35-line helper. Useful archaeology, but in the source it buries the ~6 lines of actual maths (`detector.py:781-801`) — precisely where the surviving factor-2 defect sits.
- **Six near-duplicate centred-meshgrid constructions**: `beam_stats.py:138-139` (cached), `beam_stats.py:514-516` (`M2`, uncached), `zernike.py:601-603`, `psf_mtf_otf.py:410-412`, `polychromatic.py:101-103`, `opd.py:667-669`. Only one is memoised.
- **Three inert public parameters**: `mtf_radial(..., wavelength, f)` (`:343-345`), `shack_hartmann(detector_pixels_per_lenslet=...)` (`:419-433`), `ghost_analysis(n_rays=...)`.
- **`analysis/core.py` is a pure re-export shim** (69 lines), yet `through_focus.py:54-58` imports three names through it, pulling the whole `beam_stats + strehl + psf_mtf_otf + polychromatic + zernike + opd` chain into the import graph.
- **`field.py`, `image_plane_wfe.py`, `aberration.py` are facades over `raytrace`**, sitting oddly in the wave-optics post-processing package.

---

## Unverified suspicions

- **JAX/NumPy band-limit parity in `through_focus_scan`.** NumPy uses `_get_or_make_bandlimit(...)`; the JAX kernel (`:1107-1116`) re-derives `fx_max = Lx/(2λ|z|)` inline. If the helper uses a different form (`<=` vs `<`, or Matsushima-with-sqrt) the two backends band-limit differently at the mask edge. I did not run the JAX path.
- **`polychromatic_psf` with `dy != dx`** threads `dy` into the pixel area (`:407-410`) and centroid/D4σ (`:435`) but passes only `dx` to `apply_real_lens` and `angular_spectrum_propagate` (`:381-383`) — propagation likely square-pixel while metrics are anamorphic. Not confirmed.
- **`apply_detector` has no way to turn shot noise off** (`:337-340` fires Poisson unconditionally; `seed` defaults to `None`). May be deliberate.
- **`beam_stats._centered_meshgrid` hands out shared, *writable* cache arrays** (`:122-159`) — unlike `zernike_basis_matrix`, hardened with `setflags(write=False)` in v5.29.1. No failure producible today; latent poisoning path.
- **`koehler_image`'s obliquity weight** (`coherence.py:109`) uses `cos(θ)` with `θ = √(ax²+ay²)` on a grid uniform in *angle*; the correct Jacobian to direction-cosine space is `cos(ax)·cos(ay)`. O(θ⁴); not quantified.

---

## Checked and found correct

- **Zernike indexing / normalisation / fitting**: OSA j→(n,m) and inverse verified j = 0…37 against the published table — **no 5/6 or 7/8 swap**; names match ANSI Z80.28. Gram matrix of 28 modes at N=512: max |diag−1| = 9.0e-4, max |off-diag| = 1.5e-3. Per-mode RMS = 1.0000 ± 4.5e-4. 21-mode round-trip max |Δc| = 1.06e-21 m. **Axis convention**: j=2 (Tilt X) varies along *columns*, j=1 (Tilt Y) along *rows* — row = y. `astigmatism_mag_angle` → 0°/45°/22.5°. `dy != dx` does not alias the cache.
- **`compute_psf`/`compute_otf`/`compute_mtf`**: pitch exact at oversample 1/2/4; Parseval ratio 1.000000000; PSF vs analytic Airy 1.1e-3 relative inside 10 λf/D; peak on the centre pixel; `mtf[N//2,N//2] = 1.000000000000`, `mtf[0,0] = 8.6e-18`; numerical vs analytic circular-aperture MTF 1.3e-3 for ν < 0.9; **MTF = pupil autocorrelation to 6.7e-16**.
- **Resolution metrics**: Rayleigh 2.9322 vs 2.9280 µm (+0.14 %), FWHM 2.4723 vs 2.4696 (+0.11 %), Sparrow 2.2755 vs 2.2728 (+0.12 %).
- **Strehl definitions** agree at λ/14 rms across defocus/astig/coma/spherical (0.8145/0.8180/0.8167/0.8159) vs Maréchal 0.81757.
- **`M2`**: TEM₀₀ → 1.000000 at N=256 and 512; TEM₁₀ → M²ₓ = 3.000000, M²ᵧ = 1.000000; curved Gaussian 1.000129/1.000032 — Wigner cross-term correct. `beam_d4sigma` gives exactly 2w₀.
- **`beam_diameter` / EE** on an analytic Gaussian: 1/e² 80.156 vs 80.000 µm; EE@86.47 % = 40.0125 vs 40; EE@50 % = 23.5372 vs 23.5482; curve matches `1−exp(−2r²/w²)` to 9.3e-5.
- **`apply_detector` flux conservation**: median/expected = 1.000003…1.000013 for every `(n_pixels, pitch)` in the S11-4 table; totals match the covered-area fraction to 1e-6; rectangular (32, 64) field collects 1.000005; Poisson σ/µ tracks 1/√mean.
- **`shack_hartmann` slopes**: gain 0.945–0.949 vs `centroid = f·tanθ`; flat wavefront returns bit-exact 0.
- **`wave_opd_1d` is exact** (1.06e-22 m residual); `f_ref` add-back cancels exactly. **`check_opd_sampling`'s Nyquist factor is right**: empirically correct to dx = 1.02·dx_max, failing at 1.2·dx_max, so `dx_max = λf/aperture` (factor 1, not 1/2) and the `margin >= 2` gate is appropriate. `wave_opd_2d` **with** a correct `f_ref` is exact (0.0000 waves) even 4× under-Nyquist.
- **OPD sign convention** matches CONVENTIONS.md §7 in both `wave_opd_1d/2d`.
- **`depth_of_focus`**: 4.4 µm at f/2, 550 nm = λ/(2 NA²) exactly.
- **`phase_shift_extract`**: generalised `[1, cos s, sin s]` LSQ correct for both sign conventions; reduces to the classical 4-step form.
- **`mutual_coherence`**: Γ[i,j] = ⟨E(x_i)E*(x_j)⟩ as documented.
- **`ghost_analysis`/`retrace_ghost_path`**: plate `R_i = R_j = 0.0421644` (analytic 0.0421644); `total_transmittance = 1.631072e-03` vs analytic 1.631087e-03; path enumeration correct; BSDF TIS estimator `π·mean(f)` is the correct unbiased form.
- **`polychromatic_psf` grid alignment**: ASM is pitch-preserving so all per-λ PSFs share one grid; the weighted **intensity** sum is correct and incoherent.
- **`eval_image_plane_wfe` in its valid regime**: ~3 % vs oracle, sign confirmed.
- **`error_reduction`/`hybrid_input_output`**: standard Fienup updates, correct FFT centring, convergent and correctly scaled metric.
- **`find_best_focus`**, best-focus fields, all-NaN guards, the FFT hoist — correct.
- **`petzval_radius`** sign/mirror parity, `distortion_vs_field`'s `n_obj·efl·tan θ`, `field_aberration_sweep` fan construction — all follow their cited conventions.
- **Cache hygiene**: all six caches have complete keys, locks, LRU eviction, byte caps, and registry registration. Zernike basis cache measured 2.9× (2422 → 844 ms at N=1024, 21 modes).

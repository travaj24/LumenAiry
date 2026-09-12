# TR-MAIN-2 audit — `_lens_traced.py` 9600–13227 (second half of `apply_real_lens_traced` + multi / segmented / prepared entry points)

Repo: `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, branch main.
All repro scripts under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-MAIN-2/`
(shared fixture `fixt.py`: 2 mm-aperture biconvex f≈25 mm `_TRM2_GLASS` n=1.5168 singlet, λ=1.31 µm).

## Scope read (line ranges actually read)

Line-by-line, `lumenairy/elements/_lens_traced.py`:

* 9560–9710 — exit-vertex correction, H6/C6 carrier eikonal, H3 exit-NA Nyquist guard, `_exit_na_out` fill (read for context / hand-off from TR-MAIN-1).
* 9710–10112 — coarse-grid reshape, dead-ray NaN fill, `_opl_ref` / `_opl_piston`, `_TracedExitSupport.from_landings`, the R7/P2/D1/D7/C6/C11/C12 fit-domain restriction and its arbiter/predictor, `_imap_*` domain copies.
* 10112–10270 — `inversion_method='fit'` (`_invert_fit`, `_fit_design`, hull half-planes); `use_gpu` validation; `_Cheb2DEvaluator` / `RectBivariateSpline` construction.
* 10270–10800 — paraxial-magnification stencil, `_spline_data`, `_warn_newton_unconverged`, `_invert_newton`, `_invert_newton_parallel` (pool gate, payload residency, fallbacks).
* 10798–11050 — `_support_taper`, `_ray_density_amp_grid` (Jacobian amplitude, caustic floor, fold census, entrance stop, C8 taper), `_build_newton_mask`, the inverse-characteristic gate.
* 11050–11500 — `_imap_probe_trace`, `build_inverse_map` call, the four OPL-inversion branches (backward_trace / imap-banded / imap-whole / coarse-Newton + upsample / fit / full-grid Newton), the ray-density branch matrix and its upsamples.
* 11500–11960 — `_warn_ray_density_fold`, `_origin_amp_support_verdict`, `_ray_density_self_checks` (energy / halo / retained-band).
* 11960–12450 — the **row-band assembly** (`_step3_band`, `_swap_band`, `_unit_phasor`, `_probe_band`, `_eval_band`, the two-pass ray-density band census) and the **whole-grid final assembly** (`valid`, `delta_phase`, `phase_exp`, piston phasor, masks, ray-density magnitude swap, residual multiply).
* 12452–13227 — `_carrier_reuse_key`, `apply_real_lens_traced_multi`, `_flattop_partition_1d`, `_occupied_freq_support`, `_spectral_gap_cuts`, `_segment_field_by_angle`, `apply_real_lens_traced_segmented`, `PreparedTracedLens`, `prepare_real_lens_traced`, `__all__`.

Read for dependency confirmation (outside my range): `116–167` (`_copy_prescription`), `3291–3381` (`_geometric_lens_phase`), `640–700` (pool payload residency), `7610–7767` (docstring for `caustic` / `output_plane_distance` / `_exit_na_out` / `_remap_launch_out`), `7892–8082` and `8195–8262` (amplitude_model / remap / caustic validation and the multibranch dispatch), `8330–8380` (`_chunk_assembly`), `8830–8956` (the analytic amp legs `PreparedTracedLens` must reproduce), `9129–9200` (`_imap_domain_gate`), `_lens_traced_multibranch.py:60–130, 500–756`, `_lens_imap.py:75–230, 1075–1150, 1831–1840`, `lumenairy/propagators/carrier.py:6860–6920`.

Baseline: `python -m pytest tests/unit/test_hammer_h3_traced_nyquist_guard.py tests/unit/test_hammer_h6_traced_carrier_eikonal.py -q` → **11 passed** in 204 s, so the findings below are gaps in coverage, not a broken tree.

**Not covered:** `_opl_by_backward_trace` internals (only its call site); `_Cheb2DEvaluator` / `_solve_lstsq_thread_safe` internals (TR-MAIN-1 / other partitions); `_lens_traced_uniform.py` (read only its entry contract); the GPU (`cupy`) branches — desk-checked only; `_TracedExitSupport.taper`/`retained_band_masks` internals.

---

## Findings

### [P1] Real-valued `E_in` raises `AttributeError` in the final assembly — `_lens_traced.py:11697`, consumed at `:12061`, `:12064`, `:12392`, `:12399`

```python
target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128   # 11697
...
E_out = np.where(valid, E_out, target_cdtype.type(0))                     # 12392
```

The `else` branch yields the numpy **scalar type** `np.complex128`, not a `np.dtype`. `np.dtype(...).type` exists; `np.complex128.type` does not. Every `target_cdtype.type(0)` site therefore raises. The `!=` comparisons and `.astype(target_cdtype)` calls happen to work with a bare type, which is why the bug survived — only the four `.type(0)` sites break.

Evidence (`p2_realinput.py`, numpy 2.4.6, N=256):

```
E dtype float64   iscomplexobj False
apply_real_lens        OK -> complex128
apply_real_lens_traced RAISED AttributeError :
    type object 'numpy.complex128' has no attribute 'type'    (_lens_traced.py:12392)
```

and `np.complex128.type -> AttributeError`, `np.dtype(np.complex128).type -> <class 'numpy.complex128'>`.

This is not hypothetical dead code: the sibling `apply_real_lens` accepts the same array and returns complex128, and `PreparedTracedLens.__call__` also accepts it cleanly (`p8_prepared.py`: "real input through prepared -> OK, dtype complex128"). So a caller can build a prepared screen from a real field and get an answer, but calling the element directly with the same field crashes. Both the whole-grid path (12392/12399) and the **banded path, which is the shipped default at N ≥ 4096** (12061/12064), are affected.

**Fix:** `target_cdtype = np.dtype(E_in.dtype if np.iscomplexobj(E_in) else np.complex128)` (one line; every other use of the name stays valid). Add a real-input case to the dtype tests.

---

### [P1] `caustic='multibranch'` at an axial focus returns an identically-zero field, silently — `_lens_traced.py:8209–8262` (dispatch) with `_lens_traced_multibranch.py:_ENERGY_BLOWUP_FACTOR`/`min_area_ratio`

The docstring (`:7646`, `:7663`) advertises `output_plane_distance` as the way to reach "a through-focus caustic plane … DIRECTLY (no separate ASM step)". At and around the paraxial focus the mode returns garbage, and in a ~40 µm-wide z band it returns **exactly zero with no warning at all**.

`p3_caustic.py` / `p3b_scan.py` / `p3c_dead.py`, f≈25 mm singlet, N=512, dx=4 µm, w₀=0.30 mm, paraxial BFL = 24.8341 mm (from a ray trace of the prescription), `caustic_ray_subsample=4`:

```
z (mm)     P_out / P_in      warnings
 0.0000     0.99906          0
20.0000     0.99905          0
24.0000     0.99954          0
24.7000     3.99             0          <- 4x energy GAIN, silent
24.7400     8.10             0          <- 8x energy GAIN, silent
24.7600    13.07             1
24.8000    61.76             1
24.8200     0.00             0          <- identically zero, SILENT
24.8340     0.00             0          <- identically zero, SILENT   (= BFL)
24.8600   106.59             1
26.0000     0.99914          0
```

and through the public entry point:

```
apply_real_lens_traced(..., caustic='multibranch', output_plane_distance=24.8340e-3)
   ->  P_out/P_in = 0.0000e+00,  0 non-zero pixels,  0 warnings
```

Mechanism, confirmed by a `min_area_ratio` sweep at z = 24.834 mm:

```
min_area_ratio = 1e-6 (shipped default) -> P/Pin 0.0        peak 0.0        warns 0
                 1e-8                   -> P/Pin 0.0        peak 0.0        warns 0
                 1e-10                  -> P/Pin 0.0        peak 0.0        warns 0
                 1e-12                  -> P/Pin 3.86e+09   peak 5.84e+06   warns 1
                 0                      -> P/Pin 3.86e+09   peak 5.84e+06   warns 1
```

At an axial focus every launch triangle's mapped/launch area ratio falls below `caustic_min_area_ratio`, so the rasteriser skips **all** of them and `E_flat` stays zero. The module's own D5 tripwire only fires on a power **gain** above `_ENERGY_BLOWUP_FACTOR = 10.0`, so it sees neither the total collapse nor the 1.2×–10× band (measured 4×–8× at z = 24.70–24.74 mm, silent).

Two distinct silent-failure regimes therefore exist on a mode whose headline feature is "output at `output_plane_distance`". `_lens_traced_multibranch.py`'s Scope note D5 documents the ~1e5–1e6× blow-up, but neither it nor `apply_real_lens_traced`'s docstring mentions the zero-field regime, and the caller never sees a diagnostic.

**Fix:** (a) in the multibranch/uniform kernel, count skipped-degenerate triangles and warn/raise when the skip fraction (or the non-zero pixel count inside the traced hull) collapses — "every mapped triangle was degenerate at this plane; the output plane is at/near an axial point focus, where a ring of branches coalesces and the fold-uniform swap regularises only the closest pair; use `apply_real_lens_gbd` / `apply_real_lens_fga` or single-branch `ray_density` + ASM"; (b) lower `_ENERGY_BLOWUP_FACTOR` to the documented well-behaved bound (~1.2–2×) or make the tripwire two-sided (`P/Pin < 0.5` as well); (c) add the caveat to `apply_real_lens_traced`'s `output_plane_distance` / `caustic` docstring entries, which currently read as an unqualified capability.

For reference, the wave answer on the same grid: `apply_real_lens` + `angular_spectrum_propagate(BFL)` gives P = P_in exactly, peak 73.53, EE(25 µm) = 0.6346; traced `ray_density` + ASM(BFL) gives peak 73.53, EE = 0.6346 (agreement 1e-4). The multibranch field at the same plane is 0.

---

### [P1] `newton_fit='spline'` + any vignetted (dead) ray ⇒ identically-zero field; the in-code guard comment is false — `_lens_traced.py:9710–9725`

```python
    # Dead rays would break RectBivariateSpline (which requires
    # strictly regular data); vignetting is rare for normal lenses but
    # we guard against it by filling dead entries with NaN and
    # extrapolating with the spline's natural extrapolation (OK inside
    # the entrance disc of interest).
```

`RectBivariateSpline` does **not** ignore NaN; it is an interpolating (`s=0`) FITPACK fit, so one NaN sample propagates into essentially every coefficient. Measured on a 21×21 lattice with a single NaN (`p11_vignette.py`, second block):

```
clean  ev -> [-1.3e-18  0.18  0.50]
1 NaN  ev -> [nan nan nan]
fraction of NaN spline coefficients: 0.907
```

End-to-end, with `surfaces[1]['semi_diameter'] = 0.30 mm` on the f/12.5 singlet (N=512, dx=6 µm, `ray_subsample=8`):

```
newton_fit=polynomial  P/Pin=9.99996e-01  nonzero=87253  peak=1.013
newton_fit=spline      P/Pin=0.00000e+00  nonzero=    0  peak=0.000
```

The only diagnostic the user gets (and only when `on_undersample != 'silent'`) is

```
apply_real_lens_traced Newton inversion: 1829/1829 pixels (100.0%) did not converge
to tol=6.000e-08 ... Affected pixels keep their last Newton value, which may carry
residual error.  Increase newton_max_iters if this matters ...
```

which misdiagnoses the cause (no iteration count can fix NaN coefficients) and mis-states the outcome (the field is zero, not "approximate"). `on_undersample='silent'` — used by several of the library's own tests — removes even that. `newton_fit='auto'` resolves to `'polynomial'`, whose `_Cheb2DEvaluator` does skip NaN samples, so the **default is safe**; this is the opt-in `'spline'` path. I found no `docs/audits` or `CHANGELOG.md` entry covering it.

**Fix:** on the spline branch, either (a) refuse up front when `not final.alive.all()` with a message naming vignetting and pointing at `newton_fit='polynomial'`, or (b) fill dead entries by nearest-finite extrapolation before the fit (and record the filled mask so those pixels are NaN-ed in `opl_map` afterwards), and correct the comment either way. Independently, `_warn_newton_unconverged` should distinguish "residual is NaN at every pixel" (a broken fit) from "did not reach tol" (a genuine iteration cap).

---

### [P2] `apply_real_lens_traced_multi(reuse_prepared=True)` still raises an opaque `TypeError` for 12 public kwargs — `_lens_traced.py:12660–12702` and `:12713–12716`

The v5.29 block at `:12660` states it exists to replace "the opaque `TypeError: prepare_real_lens_traced() got an unexpected keyword argument` they used to raise from three frames down — and only on the DEFAULT reuse path, so the same call worked or crashed depending on the carrier kind." The fix enumerated six keys in `_NO_SCREEN`. `prepare_real_lens_traced`'s signature is missing **twelve** more of `apply_real_lens_traced`'s public kwargs, all of which `_multi` forwards verbatim:

```
dy, remap_sampling, on_fit_domain_basis, on_pool_memory, parallel_amp_min_free_gb,
newton_mask_dilate_coarse_px, fast_analytic_phase, output_plane_distance,
caustic_ray_subsample, caustic_band, caustic_min_area_ratio, origin
```

Measured (`p6b_multi_kwargs.py`, N=512, two emitters, `carriers=None`):

```
on_pool_memory               reuse=True  TypeError: prepare_real_lens_traced() got an unexpected keyword argument 'on_pool_memory'
on_pool_memory               reuse=False OK
newton_mask_dilate_coarse_px reuse=True  TypeError ...
newton_mask_dilate_coarse_px reuse=False OK
dy                           reuse=True  TypeError: ... 'dy'. Did you mean 'dx'?
dy                           reuse=False OK
origin                       reuse=True  TypeError ...
origin                       reuse=False OK
```

i.e. exactly the failure mode the comment claims was closed, with the same carrier-kind-dependent reachability (`carriers='auto'` or an ndarray takes the per-emitter path and works).

**Fix:** derive the accepted set programmatically instead of by hand — compare `inspect.signature(apply_real_lens_traced)` against `inspect.signature(prepare_real_lens_traced)` at import time (or in a test) and route any kwarg the prepared factory cannot take into `_NO_SCREEN`-style rejection with a reason; alternatively give `prepare_real_lens_traced` a `**traced_kwargs` pass-through with an explicit input-dependence deny-list.

---

### [P2] `apply_real_lens_traced_segmented` silently bypasses the square-pixel refusal — `_lens_traced.py:12885`, `:12952–12972`

`apply_real_lens_traced_segmented` takes `dy` and hands it to `_segment_field_by_angle(E_in, dx, dy, ...)`, so the angular partition uses the caller's `dy`. It then calls `apply_real_lens_traced(..., dx=dx)` **without** `dy`, so the traced passes run with `dy = dx`. The element's own anamorphic-grid refusal is therefore never reached:

```
segmented(dy=2*dx)      -> returned OK, shape (512, 512)
direct traced(dy=2*dx)  -> ValueError: apply_real_lens_traced currently requires square pixels (dx == dy)
```
(`p12_misc.py`.)

The result is internally inconsistent — the fy axis of the spectral split is scaled by `dy` while the ray trace, the exit grid and the aperture mask all assume `dy = dx` — and the caller gets no signal.

**Fix:** raise the same square-pixel `ValueError` at the top of `apply_real_lens_traced_segmented` (before segmenting), or forward `dy=dy` so the element raises.

---

### [P2] `_segment_field_by_angle` materialises the separable spectral windows as full 2-D grids — `_lens_traced.py:12864–12873`

```python
FX, FY = np.meshgrid(fx, fy)
Wx = _flattop_partition_1d(FX, cutx, hwx)     # Kx full (Ny, Nx) float64 grids
Wy = _flattop_partition_1d(FY, cuty, hwy)     # Ky full (Ny, Nx) float64 grids
```

`_flattop_partition_1d` is a function of one coordinate only, so `Wx[i]` is constant along rows and `Wy[j]` along columns. Verified bit-exactly: `max|Wx[0] - broadcast(Wx1d[0][None, :])| = 0.000e+00` where `Wx1d = _flattop_partition_1d(fx, cutx, hwx)` (`p15_segmem.py`).

Measured on a two-beam ±25 mrad field:

```
N= 512  nseg=13  1.061 s  peak 102.8 MB = 49.0 full float64 grids
        windows: 2-D form 13 x 2.10 MB = 27.3 MB ; 1-D form 13 x 4.1 kB
N=1024  nseg=15  5.188 s  peak 461.4 MB = 55.0 full float64 grids
        windows: 2-D form 15 x 8.39 MB = 125.8 MB ; 1-D form 15 x 8.2 kB
```

**Fix:** build the partitions on the 1-D axes and form the product lazily —
`Wx = _flattop_partition_1d(fx, cutx, hwx)`, `Wy = _flattop_partition_1d(fy, cuty, hwy)`, then `(wi[None, :] * wj[:, None]) * F` — and drop the `FX, FY = np.meshgrid(...)` pair entirely. Saves `(Kx + Ky + 2)` full float64 grids (126 MB at N=1024, ~2 GB at N=4096) for bit-identical output.

Related, same function: the `min_segment_power` filter is applied **after** `np.fft.ifft2`, so every one of the `Kx*Ky` bins pays a full inverse FFT even when it carries no power. On the two-beam case 13 of 15 bins hold < 1e-3 of the power. Testing the in-band power by Parseval on `(wi*wj)*F` before transforming would skip those FFTs (~7× less FFT work here) with no change to the result.

---

### [P2] `inversion_method='fit'` at `ray_subsample=1` builds the full-grid Chebyshev design matrix unchunked — `_lens_traced.py:10147–10190` (`_fit_design`, `_invert_fit`) called at `:11465`

`_invert_fit(X, Y)` on the whole wave grid evaluates `chebvander` twice at `(N², order+1)` and then the total-degree product at `(N², M)` with `M = (order+1)(order+2)/2 = 28` at the default `newton_poly_order=6`. Nothing chunks it, although the module already has `_CHEB_FIT_CHUNK_ENTRIES = 8_000_000` for exactly this purpose (used by `_Cheb2DEvaluator` at `:3038`) and `_TracedExitSupport.signed_distance` right below is chunked.

Measured peak (`p14_fit.py`, tracemalloc, units of one full float64 grid `8N²`):

```
N= 512 fit sub=1   9.14 s  peak  414.4 MB = 197.6 grids   (analytic leg alone 41.6)
N= 512 fit sub=8   2.20 s  peak   63.5 MB =  30.3 grids
N=1024 fit sub=1  19.99 s  peak 1183.0 MB = 141.0 grids   (analytic leg alone 24.1)
N=1024 fit sub=8   1.91 s  peak  227.6 MB =  27.1 grids
```

141 full-grid units at N=1024 projects to ~76 GB at N=8192 and ~300 GB at N=16384 — an OOM on the very grids this file's memory work targets.

**Fix:** loop `_invert_fit` over output blocks of `_CHEB_FIT_CHUNK_ENTRIES // M` entries, writing into a preallocated `(N, N)` output. Bit-identical (the design is pointwise in the output) and caps the peak at one output grid plus one bounded block.

---

### [P2] Whole-grid final masking allocates ~7 full-grid units where ~3 suffice — `_lens_traced.py:12392`, `:12397–12399`

```python
E_out = np.where(valid, E_out, target_cdtype.type(0))
...
E_out = np.where(X ** 2 + Y ** 2 <= (aperture / 2) ** 2, E_out, target_cdtype.type(0))
```

`X`/`Y` are `np.broadcast_to` views of the 1-D axes (a deliberate memory choice at `:8380`), so `X ** 2` and `Y ** 2` each materialise a full `(N, N)` float64 array and their sum a third; each `np.where` allocates a fresh complex output while the old one is still live. Counting in units of `8N²` bytes (complex128 = 2 units), the peak across these two statements is ≈ 2 (`E_out`) + 2 (new `E_out`) + 3 (the radius temporaries) + 0.125 (bool) ≈ **7.1 units** ≈ 61 GB at N = 32768 — for two masking operations.

The banded path at `:12062–12065` already uses the cheap idiom (`x[None, :] ** 2 + _y_ax[r0:r1, None] ** 2`, one temporary).

**Fix:** in-place masking plus the axes form:

```python
np.copyto(E_out, 0, where=~valid)
if aperture is not None:
    _r2 = x[None, :] ** 2 + _y_ax[:, None] ** 2        # 1 temporary, broadcast
    np.copyto(E_out, 0, where=_r2 > (aperture / 2) ** 2)
```

≈ 3.1 units instead of 7.1 — a 2.3× lower peak at that stage, bit-identical.

---

### [P2] `_ray_density_self_checks` upcasts the whole input field and builds full-grid power temporaries — `_lens_traced.py:11792`, `:11801` (and the halo/band checks at `:11854`, `:11914`)

```python
_rd_pin  = np.abs(np.asarray(E_in, dtype=np.complex128)) ** 2     # 11792
_rd_p_out = float((np.abs(E_out) ** 2).sum())                     # 11801
```

For a **complex64** input, `np.asarray(E_in, dtype=np.complex128)` copies the entire field to complex128 purely to take its modulus. Measured at N=512 (`p12_misc.py`):

```
np.abs(np.asarray(E64c, complex128)) ** 2   peak 6.29 MB   (field 2.10 MB)   -> 3.0x field
np.abs(E).astype(np.float64); b *= b        peak 3.15 MB                     -> 1.5x field
(np.abs(Z) ** 2).sum()                      peak 4.19 MB
float(np.vdot(Z, Z).real)                   peak 0.0003 MB   (agree to 2.2e-16)
```

At N = 32768 with a complex64 field that is ~25.8 GB vs ~12.9 GB for the first, and ~17.2 GB vs ~0 for the power sum. This is a *diagnostic* stage running after the field is finished, i.e. exactly where the peak plateau lives.

**Fix:** `_rd_pin = np.abs(E_in).astype(np.float64); _rd_pin *= _rd_pin` (drop the complex128 cast entirely), and `_rd_p_out = float(np.vdot(E_out, E_out).real)` — measured agreement 2.2e-16 relative. Same for the `(np.abs(...) ** 2).sum()` inside the halo and support-band checks.

Related (smaller): on the whole-grid ray-density swap, `E_out = _ard * _unit` multiplies a float64 magnitude by a complex64 phasor, producing a **complex128** full grid that is immediately cast back to complex64 at `:12446` — one extra 2-unit grid on a complex64 request. `_ard.astype(_unit.real.dtype, copy=False)` before the multiply avoids it.

---

### [P3] `_spectral_gap_cuts` docstring contradicts the code on where flanking peaks are looked for — `_lens_traced.py:12771–12791`

The docstring: "*it is flanked (within the occupied ``[lo, hi]``) by peaks above ``peak_frac``*". The code:

```python
if p[:i].max() > peak_frac and p[i + 1:].max() > peak_frac:
```

scans the **whole** marginal, not the occupied band. A cut can therefore be justified by a peak outside the 0.995-power support (i.e. by spectral leakage/noise). Observed on a clean two-beam ±25 mrad field: the true gap is a single cut at f = 0, but the function returns three cuts `[-14323, 0, +14323] m⁻¹` (`p7b_cuts.py`), carving two empty bins. Harmless at the default `min_segment_power=1e-3` (empty bins are dropped) but it costs two extra inverse FFTs at `min_segment_power=0`, the setting the "segments sum to the input EXACTLY" contract requires.

**Fix:** restrict both maxima to `inband` (`p[idx[idx < i]]` / `p[idx[idx > i]]`), or correct the docstring.

### [P3] Single-segment path forwards a possibly-sequence `carriers` as a scalar `carrier` — `_lens_traced.py:12965–12969`

```python
if len(segments) == 1:
    return apply_real_lens_traced(segments[0], ..., carrier=carriers, **traced_kwargs)
```

`carriers` may legitimately be a per-segment list (the `_multi` branch below accepts one). The segment count is **data-dependent**, so the same call is valid or ill-typed depending on the input spectrum. Either reject sequence `carriers` when one segment results, or take `carriers[0]`, and document that a per-segment list cannot be supplied for an `'auto'` split whose count the caller cannot predict.

### [P3] `output_plane_distance` / `caustic` docs promise a capability the axial-focus case does not deliver — `_lens_traced.py:7646–7648`, `:7663–7667`

"*Output is taken at ``output_plane_distance`` past the exit vertex, so a through-focus caustic plane is reached DIRECTLY (no separate ASM step).*" The Scope caveat immediately below is about the missing dark-side diffraction tail; it does not say that at an on-axis focus the answer is zero or 1e2–1e9× the input power (see the P1 above). Add the caveat here, not only in `_lens_traced_multibranch.py`'s internal Scope note D5.

---

## Performance opportunities

Ranked by measured or projected gain. All numbers from `p10_mem.py`, `p10b_prof.py`, `p14_fit.py`, `p15_segmem.py` on this box (Windows 11, CPython 3.14, numpy 2.4.6).

1. **`inversion_method='fit'` chunking** — 141 → ~3 full-grid units at N = 1024 (1183 MB → ~30 MB); makes the mode usable above N ≈ 4096 at all. See the P2 above.
2. **Segmented spectral windows on 1-D axes** — removes `(Kx + Ky + 2)` full float64 grids; 126 MB of 461 MB at N = 1024, ~2 GB at N = 4096.
3. **Whole-grid final masking in place** — peak at that stage 7.1 → 3.1 full-grid units (61 → 27 GB at N = 32768).
4. **`_ray_density_self_checks` allocation** — ~17 GB of transients removed at N = 32768 (`np.vdot` instead of `(np.abs**2).sum()`; drop the complex128 upcast of `E_in`).
5. **Segmented: Parseval pre-filter before `ifft2`** — on the measured two-beam case 13 of 15 inverse FFTs are for bins below `min_segment_power`; skipping them is ~7× less FFT work in `_segment_field_by_angle`.
6. **The inverse-characteristic (`inverse_map`, shipped `True`) is a large win at the shipped `ray_subsample`, and a loss at small N with small `sub`.** Measured, f/12.5 singlet, warm imap cache:

   | case | imap default | `inverse_map=False` |
   |---|---|---|
   | N=512, sub=8, screen (cold) | 1.83 s | 0.85 s |
   | N=512, sub=8, screen (warm) | 0.71 s | 0.69 s |
   | N=1024, sub=4, screen | 2.82 s | 3.71 s |
   | N=1024, sub=4, ray_density | 4.65 s | **11.98 s** (2.6×) |

   and it is *more accurate*: RMS phase difference between the two inversions at N=512/sub=8 is 7.73e-02 rad (1.23e-02 waves, max 0.112 rad), and the record reports the model's own held-out OPL error at 2.08e-12 waves against the incumbent's 3.90e-09 — i.e. the difference is the coarse-lattice upsample, which the model removes. Ray-density energy confirms it (`p4_energy.py`, ratio to aperture-transmitted input power):

   ```
   sub=  1  2  4  8   imap=True :  0.999934 0.999934 0.999934 0.999934
   sub=  1  2  4  8   imap=False:  0.999934 0.999703 0.998780 0.995106
   ```

   Nothing to fix; worth noting that the build is ~0.8 s and is content-cached, so the cold-call regression at small N is real but one-shot.
7. **The assembly is *not* the memory bottleneck at moderate N.** At N = 1024, `ray_subsample=4`, whole-grid: total peak 244.4 MB of which `apply_real_lens` alone is 239.0 MB. The v5.16.2 / v5.44 lifetime work holds up; items 3–4 above are the remaining assembly-side slack and they only matter at the 16k–32k grids.

---

## Alternative algorithms / methods

* **Replace the blind spectral segmentation with a Gabor / Gaussian-beam decomposition.** `_segment_field_by_angle` splits only where the *marginal* 1-D angular spectra have deep valleys; it cannot separate two congruences that overlap in `fx` and in `fy` but not jointly (a separable cut in a non-separable problem), and it is silent when it fails. The library already ships `lumenairy/propagators/gbd.py` and `elements/lenses_gbd.py`. A Gabor frame decomposition (Bastiaans, *Proc. IEEE* 68, 538 (1980); Einziger, Raz & Shapira, *JOSA A* 3, 508 (1986)) gives a *guaranteed complete* expansion into elementary beams, each of which is a single congruence by construction, so the "one ray congruence per output pixel" premise is satisfied exactly rather than heuristically. Cost: O(K) traced passes with K ≫ the current 1–32, but each on a much smaller sub-aperture — and the current `min_segment_power` truncation already abandons exactness, so the honest comparison is "K beams, controlled error" against "2–32 bins, uncontrolled error". A cheap intermediate: partition in the joint `(fx, fy)` plane by watershed on `P` rather than by two independent marginal splits, which would fix the non-separable case at almost no extra cost.
* **The K-carrier decomposition the docstring names as future work (`:6810`) is already implemented for the known-emitter case** — `apply_real_lens_traced_multi` is exactly `Σ_k traced(E_k, carrier_k)`, and my measurement (below) shows the per-branch pistons are consistent to 0.3 mrad. The missing half is the *blind* K-carrier fit; a Gabor/ Gaussian-beam front end (above) supplies the per-branch carriers for free (each beamlet's own centroid direction), which is a shorter path than improving `_spectral_gap_cuts`.
* **Caustic band: use the library's own Maslov propagator instead of the ART branch sum.** `elements/lenses_maslov.py` (`apply_real_lens_maslov`, with its own `output_plane_distance`) is a phase-space propagator that is caustic-*safe* by construction — the mixed position/momentum representation has no `1/√|det J|` singularity (Maslov & Fedoriuk, *Semi-Classical Approximation in Quantum Mechanics*, Reidel 1981; Kravtsov & Orlov, *Caustics, Catastrophes and Wave Fields*, Springer 1999, ch. 5). Given the P1 above (zero field at the focus and 1e9× energy one triangle away), routing `caustic='multibranch', output_plane_distance≈BFL` *through* the Maslov propagator when the fold degenerates to an axial point focus would turn a silent catastrophe into a correct answer using code that already exists.
* **Pearcey / higher catastrophes for the cusp.** `ludwig_fold` (Ludwig, *Comm. Pure Appl. Math.* 19, 215 (1966); Kravtsov, *Sov. Radiophys.* 7, 664 (1964)) regularises a **fold** (two coalescing branches, Airy). At an axial point focus of a rotationally symmetric system a whole *ring* coalesces — a higher catastrophe — and the pair swap provably cannot regularise it, which is the root cause of the P1. The uniform functions for the next cases are the Pearcey integral for the cusp (Pearcey, *Phil. Mag.* 37, 311 (1946); Connor & Farrelly, *J. Chem. Phys.* 75, 2831 (1981), for the numerics) and, for the rotationally symmetric focus specifically, the Bessoid / axial-caustic uniform approximation (Kirk, Wells & Berry, *Phys. Rev. Lett.* 92, 143002 (2004)). Any of these is a substantial build; the pragmatic fix remains "detect and refuse", which is what the P1 recommendation asks for.
* **Newton-free inversion at `sub = 1`.** The `inversion_method='fit'` path already fits the inverse characteristic directly in exit coordinates; `_lens_imap`'s degree-14 model does the same thing better and is content-cached. The 'fit' path's remaining reason to exist is the spline-free, Newton-free evaluation, and its unchunked design matrix (P2 above) is what makes it unusable at scale. Chunking it would also let the gate `_imap_domain_gate`'s `sub > 1` restriction be revisited: at `sub = 1` the model is currently refused on the grounds that "the Newton already runs per pixel", but a cached degree-14 evaluation is cheaper than a 12-iteration Newton per pixel.

---

## Code organization observations

* `apply_real_lens_traced` is **5722 lines** (6731–12452) in one function body, with ~570 lines of docstring and, by my count in this half alone, 14 named niche/audit tags (C1, C4, C6, C7, C8, C11, C12, C14, C15, D1, D7, D9, K3, N12/P11, S12) whose invariants cross-reference one another. My half contains nine nested closures (`_warn_newton_unconverged`, `_invert_newton`, `_invert_newton_parallel`, `_support_taper`, `_ray_density_amp_grid`, `_build_newton_mask`, `_imap_incumbent`, `_imap_probe_trace`, `_warn_ray_density_fold`, `_origin_amp_support_verdict`, `_ray_density_self_checks`, `_probe_band`, `_step3_band`, `_swap_band`, `_unit_phasor`, `_eval_band`) that close over ~40 free variables. The `# noqa: F821` at `:12058` with a hand-maintained line-number comment ("*calls at [12093, 12223], del at [12350]*", re-checked 2026-09-11) is the clearest symptom: correctness of a `del` now depends on a comment that has to be re-verified by hand after every edit. The natural seam is a `_TracedAssembly` dataclass holding the resolved geometry + masks, with `step3`, `ray_density_swap` and `self_checks` as methods — the band and whole-grid paths then share one implementation instead of two textually-parallel copies.
* **The band and whole-grid paths are duplicated expression-for-expression** in five places (Step 3, the aperture mask, the ray-density swap, the residual transport, the probe gather), kept in sync by a byte-identity test. I re-verified that identity independently (see below) and it holds today, but the duplication is the thing that made the v5.44.1 D7 probe bug possible (the probe block sat after the band path's `return`).
* `_carrier_reuse_key`, `_flattop_partition_1d`, `_occupied_freq_support`, `_spectral_gap_cuts` and `_segment_field_by_angle` are clean, short, and testable — they read like a different (better) file. They are also the only part of my range with no audit-tag comments, which is probably not a coincidence.
* `prepare_real_lens_traced` re-declares 23 of `apply_real_lens_traced`'s parameters by hand. That hand-maintained duplication *is* the P2 above.
* Comment/code drift found: the `RectBivariateSpline` NaN guard (P1, `:9711`), the `_spectral_gap_cuts` docstring (P3, `:12775`), and `PreparedTracedLens`'s "*same 8 kwargs*" comment at `:13044` which now describes 8 kwargs but is checked by nothing — a signature change in the internal `_amp_call` (`:8861`) would desynchronise the prepared screen from its amplitude leg with no test failing except by luck.

---

## Unverified suspicions

* **`_exit_na_out` fails open.** It is filled only inside `if _sig.any():` (`:9667`), and the multibranch dispatch returns at `:8262` before it is reached at all. `carrier.py:6892` guards with `float(_na_diag.get('na_exit') or 0.0) > 0.0`, so an unfilled dict silently *skips* the `on_tilt_exact_grid` refusal whose default is `'error'`. I could not construct a realistic field where `_sig.any()` is False (a zero field makes `_amp >= 0` true everywhere), so I cannot show it bites. Confirming would need a chain-level call where the element early-returns or every ray dies.
* **`_geometric_lens_phase` under `set_default_real_dtype('float32')`** (`:3348`) accumulates `k0·n·t ≈ 7.5e4` rad in float32 before wrapping — ulp ≈ 0.009 rad, i.e. ~λ/700 of phase error injected into `phase_analytic_lens` on the `fast_analytic_phase=True` path. Below the λ/100 bar the consumers state, so probably acceptable, but it is an unguarded interaction between two opt-in knobs and I did not measure it end-to-end.
* **`_POOL_RESIDENT_PAYLOAD_KEY`** (`:10584`, `:10704`, `:10737`) is read and written outside `_PERSISTENT_POOL_LOCK`. I believe this is benign — the key is a content digest, a wrong belief raises `NewtonPayloadNotResident` and the parent re-sends — but I did not build a two-thread stress test to prove no interleaving can produce a *silent* wrong answer.
* **`min_segment_power` and `max_segments` interaction:** if every bin falls below `min_segment_power` the code falls back to `[E.copy()]` (`:12874`), silently discarding the partition. I could not produce that state from a physical field, but with `max_segments=32` and a broad multi-lobed spectrum it looks reachable.

---

## Checked and found correct (brief)

* **Final phase assembly / dtype ordering (probe 1).** `np.exp(1j·Δφ)` is evaluated in complex128 and *then* cast to `target_cdtype` (`:12365–12370`, `:12044–12048`), and `_opl_piston_phasor = np.exp(1j·k0·_opl_piston)` is a float64 scalar exponential, so the ~5e4 rad absolute piston never reaches a float32 exponential. Measured at N=512 on a 5 mm-thick element: complex64 vs complex128 output differs by max 3.89e-07 / rms 3.33e-08 of peak; core phase difference max 2.03e-06 rad, rms 2.47e-07 rad; power 6981.28984 vs 6981.28842. `phase_analytic_lens` is `np.angle(...)`, i.e. already wrapped. No float32 phase-wrap defect.
* **Masked / non-converged pixels (probe 2).** They get *exactly zero*, not NaN: `valid = np.isfinite(opl_map)` gates both `Δφ` and the field, and the ray-density branch zeroes through `np.where(np.isfinite(ard_map), ard_map, 0.0)` and a unit phasor that is 0 where the screen is 0. There is **no `np.nan_to_num`** anywhere in the file. Measured over top-hat (inside and over-filling the aperture) and Gaussian inputs × `screen`/`ray_density`: NaN = 0, Inf = 0 in every case; `P_out/P_in` = 0.999991 / 0.998063 (r = 0.9 mm top hat), 0.508634 / 0.510199 (r = 1.4 mm top hat past the 1.0 mm stop — the expected area ratio 0.510), 0.999996 / 0.999930 (Gaussian). The support-band self-check correctly fired on the over-filled top hat.
* **`amplitude_model` enumeration and energy (probe 4).** Exactly two values, `'screen'` and `'ray_density'` (`:7899`), validated with a message. Energy against the aperture-transmitted input power: `screen` 1.000000 at every `ray_subsample ∈ {1,2,4,8,16}`; `ray_density` 0.999934 (sub 1–8, shipped inverse-map path), 0.980679 at sub=16. No self-check warning fired in any of the 20 cells.
* **Coherent sum in `apply_real_lens_traced_multi` (probe 6).** `E_out = Σ_k contrib_k` is a plain sum and it is *exactly* `T1 + T2` (relative difference 0.0). Analytic linearity holds to 5.3e-16. Per-branch **piston consistency** — the thing the FIX_TILT_QUADRATIC_OPL_2026_08_11 piston restoration exists for — verified on an *asymmetric* pair (0 mrad and 12 mrad `TiltedCarrier` branches, which have genuinely different tilt-quadratic pistons): the interference fringe phase `arg(T1) − arg(T2)` matches the analytic reference to **mean +0.0003 rad, rms 0.0004 rad** over 12373 overlap pixels. The coherent sum is physically meaningful. (Symmetric ±5 mrad: `multi(auto)` vs `analytic(sum)` 1.09e-02, `multi(None)` 1.61e-03 = the same as `traced(sum)` — i.e. the extra 1e-2 is the per-emitter `'auto'` carrier-fit residual the docstring warns about, not a piston error.) `TiltedCarrier` is correctly matched *before* the sequence test (`:12603`).
* **Spectral partition of unity (probe 7).** `_flattop_partition_1d` sums to 1 to machine precision for every configuration tried (`max|Σ−1| = 2.22e-16` for 1, 2 and 3 cuts, including `hw` larger than half the band); `_hw` (`:12852–12860`) correctly keeps `2·hw < min gap`. End-to-end, `apply_real_lens_traced_segmented(..., min_segment_power=0, return_segments=True)` reconstructs the input to `max|Σ segments − E| / max|E| = 3.5e-16` (two-beam), `5.7e-16` (single Gaussian), `0` (all-zero field). At the default `min_segment_power=1e-3` the reconstruction error is 2.7e-05, as documented. Degenerate inputs behave: all-zero power → full band and no cuts; a single peak → no cuts (correctly one segment = plain traced); no division by zero (`tot_power + 1e-300`, `pk <= 0` early return). Zero-length arrays raise, but cannot arise from a validated 2-D field.
* **`PreparedTracedLens` / `prepare_real_lens_traced` (probe 8).** The screen factorisation is **exact**: `prep(E)` vs the equivalent direct call gives relative difference **0.0** for a matching Gaussian, a 2.7× smaller Gaussian, an 8 mrad-tilted field and a top hat (i.e. the screen genuinely is input-independent across amplitude footprint and tilt), and 1.2e-07 for complex64 (float32 rounding only; the prepared path multiplies in complex128 and rounds once, the direct path rounds the phasor first — the prepared result is if anything the more accurate). Shape mismatch raises with a clear message; `release()`/`clear()` is idempotent and the post-release call raises. Staleness: `_copy_prescription` deep-copies, so an in-place `surfaces[0]['radius']` edit after preparing leaves the prepared object self-consistently on the *old* lens (measured 1.055 relative difference against a rebuild) — the documented E-H4 behaviour, and both halves of the prepared object (screen and amplitude leg) use the same frozen copy. `wave_propagator` / `sag_dtype` are resolved and stored. `'auto'` carriers, `amplitude_model='ray_density'` and `fit_radius_beam_factor` are rejected with reasons. There is **no module-level prepared cache**, so there is no stale-cache surface here; `_multi`'s `prepared_cache` is per-call and keyed by `_carrier_reuse_key`, which correctly returns `None` (never share) for `'auto'` and for ndarray wavefronts and a value tuple for `TiltedCarrier`/scalars. The only residual aliasing risk is `_copy_prescription`'s deepcopy-failure fallback (`:132–143`), which shares leaf ndarrays such as `form_error` — narrow, and explicitly a deliberate trade.
* **v5.44 banded assembly is byte-identical to the whole-grid path.** Independently re-verified (`p13_band.py`, N=512, `sub=8`, `sag_chunk_rows=0` vs `64`) across all seven configurations: screen/ray_density × `inverse_map` True/False, plus `preserve_input_phase='remap'` with `remap_sampling` `'lattice'` and `'full'`. `np.array_equal` **True** in every case (`max|d| = 0.000e+00`). The two-pass ray-density band census (median/min/max/sign of `|det J|`, including the cross-band sign scan at `:12223–12228`) reproduces the whole-grid census exactly.
* **`inverse_map` coupling (probe 5).** `_IMAP.imap_enabled(inverse_map)` resolves `None` → module `TRACED_INVERSE_MAP = True`; the gate is `sub > 1 and inversion_method == 'newton' and not use_gpu` (`:9181`, `:11044`), so the per-pixel inverse characteristic *is* the shipped default at `ray_subsample=8` and it replaces the coarse Newton + `map_coordinates` upsample chain entirely (`:11195–11273`). The build cache key (`_lens_imap.py:1092–1123`) hashes the **bytes** of every array that enters the fit plus the version, the flags, the incumbent's evaluation fingerprint and the `parity_tag` — I could not construct a stale hit. Timings and accuracy in the performance section.
* **Newton pool payload residency** (`:10687–10737`, `:663–695`): content-addressed key (blake2b of the exact pickle), at most one resident payload per worker, `NewtonPayloadNotResident` retry path. A wrong residency belief cannot produce a wrong answer, only a round trip.
* **No `warnings.simplefilter` / `filterwarnings` calls anywhere in the file** (only a doc reference at `:11782`), so no global warning-filter leak. No bare `except Exception`. The v5.44.1 `stacklevel=3` fix on the three self-check warnings is consistent with its two sibling closures.
* **`caustic` validation and routing** (`:7943–7974`, `:8068–8074`, `:8209–8262`): `caustic='single'`/`None` are byte-identical; `'multibranch'`/`'uniform'` correctly require `amplitude_model='ray_density'` and the CPU path, correctly refuse a scalar/ndarray carrier and a non-zero `origin`, and `output_plane_distance != 0` correctly raises on every non-multibranch mode. Away from the focus the multibranch field is sound: at the exit vertex `P/P_in = 0.99906` with peak 1.0134 against the analytic 1.0269, and at z = 20 mm it tracks a converging beam correctly (`P/P_in = 0.99905`).
* **Probe 3's traced-vs-wave comparison on a fast singlet** (f/4.2 biconvex, 87 µm of longitudinal spherical aberration): exit-plane `|T − A|/|A| = 1.06e-01` (the traced OPD refinement) with `P_T/P_A = 0.999993`; after ASM to three planes through focus the peak intensities differ by up to 12 % and EE(10 µm) by up to 0.009, with identical peak positions. That disagreement is **physical** (the analytic split-step carries the paraxial thin-surface OPD; the traced model carries the exact ray OPD), not a bug.

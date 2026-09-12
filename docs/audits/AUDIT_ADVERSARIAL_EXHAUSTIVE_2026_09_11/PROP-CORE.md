# PROP-CORE audit — core propagator kernels, FFT infrastructure, dispatcher, system chain, backend shims

All measurements below were run on this box with CPython 3.14 / numpy 2.4.6 / scipy 1.17.1 / pyfftw 0.15.1.
Repro scripts: `…/scratchpad/PROP-CORE/p1_asm_core.py` … `p11_fresnel_alias.py`.
**Timing caveat:** ~20 other audit processes were running concurrently; wall-clock numbers are indicative, memory (tracemalloc) and accuracy numbers are not load-sensitive.

## Scope read (line by line)

| File | Read |
|---|---|
| `lumenairy/propagators/asm.py` | 1–1264 (all) |
| `lumenairy/propagators/fresnel.py` | 1–515 (all) |
| `lumenairy/propagators/rs.py` | 1–370 (all) |
| `lumenairy/propagators/sas.py` | 1–352 (all) |
| `lumenairy/propagators/_bluestein.py` | 1–606 (all) |
| `lumenairy/propagators/fft_infra.py` | 1–300, 976–1360, 1355–1800, 1938–2230 (all executable paths; the ~300 lines of pure prose comment blocks at 300–614 and 614–976 skimmed for setter semantics) |
| `lumenairy/propagators/mft.py` | 50–170, 300–620 (`angular_spectrum_propagate_mft`, `resample_field`, validators); `fresnel_propagate_mft` / `fraunhofer_propagate_mft` outlines + their Bluestein call sites |
| `lumenairy/propagators/dispatch.py` | 1–60, 110–135, 380–560, 787–1000, 1094–1340 (selection + dispatch); the ~300 lines of docstring at 135–380 skimmed |
| `lumenairy/propagators/system.py` | 56–230, 660–990 (element loop), 1426–1520 (JAX kernel signatures) |
| `lumenairy/propagators/ensemble.py` | 260–440 (accumulator) |
| `lumenairy/propagators/subaperture.py` | 190–450 (`combine_patch_fields`, ABCD mapping) |
| `lumenairy/backend/array.py`, `fft.py`, `scipy.py`, `random.py` | structural scan of every `def`/`return`/dispatch line + full read of `scipy.py:74–125` and `random.py:1–130` |
| `lumenairy/elements/_lens_real.py:1868–1912` | `_propagate_through_glass` (dependency follow, to answer probe 1a) |

Also read after the first pass: `lumenairy/propagators/vector_diffraction.py` 60–477 (all of `richards_wolf_focus` + `debye_wolf_psf`) and `result.py` 60–219 (all of `PropagationResult`).

**Not reached:** `hf.py` (687), `hfpi.py` (1143), `vectorial_hfpi.py` (474), `mhs.py` (708) — 3012 lines delegated to a subagent that had not returned when this report was written, so that sub-partition is **uncovered**; `propagators/__init__.py` beyond `__all__` verification; `subaperture.py` lines 1–190 and 450–614; the ~800 comment-only lines of `fft_infra.py` at 300–976.

---

## Findings

### [P1] `lumenairy.backend.scipy.jv(v, x)` computes `scipy.special.jv(x, v)` — arguments swapped
`lumenairy/backend/scipy.py:109` (with `:74–98`)

```python
def jv(v, x):
    if is_jax_array(x):
        return _jax_special.bessel_jv(v, x)      # CORRECT order
    return _dispatch_special('jv', x, v)          # -> _sp_special.jv(x, v)   SWAPPED
```
`_dispatch_special(name, x, *args)` ends in `getattr(_sp_special, name)(x, *args)`, i.e. `scipy.special.jv(x, v)`, but scipy's signature is `jv(v, z)` — order first, argument second. The JAX branch three lines above uses the correct order, so the two backends disagree.

**Measured**
```
jv(v=0,   x=2.0): backend = +0.000000000   scipy.jv(0,2)   = +0.223890779
jv(v=1,   x=3.0): backend = +0.019563354   scipy.jv(1,3)   = +0.339058959
jv(v=2,   x=1.5): backend = +0.491293779   scipy.jv(2,1.5) = +0.232087672
jv(v=0.5, x=4.0): backend = +0.000160736   scipy.jv(.5,4)  = -0.301920513
jv(1, array([1,2,3])) = [0.44005 0.11490 0.01956]  expected [0.44005 0.57672 0.33906]
```
Note the array case's first element is correct by coincidence (`J₁(1)`), which is exactly the kind of value a smoke test would check.

**Impact** — no current in-library consumer: `elements/bor/farfield.py:28`, `bor/fiber_oracle.py:25` and `bor/stepindex_oracle.py:31` all import `jv` from `scipy.special` directly. So this is latent, not live. But it is a module-level function in the public backend package, it silently returns a *plausible* wrong number, and the JAX branch is right — anyone who wires it up for a GPU/JAX path gets numpy-vs-jax divergence. Prior audits touched `backend/scipy.py` only for import time (`AUDIT_V5_17_0_2026_07_01_DEEP.md`); the argument order is not mentioned anywhere in `docs/audits` or `CHANGELOG.md`.

**Fix** — `_dispatch_special` hard-codes "the array is the first argument", which is wrong for two-argument special functions. Either give it an argument position, or special-case `jv`:
```python
if is_cupy_array(x):
    ... _cu_special.jv(v, x) / host round-trip with (v, x_host) ...
return _sp_special.jv(v, x)
```
Repro: `p10.py`, and the inline `python -c` in the transcript.

---

### [P1] `fresnel_propagate` has no chirp-sampling guard — silently ~33 % wrong for `z < N·dx²/λ`
`lumenairy/propagators/fresnel.py:174–358`

The single-FFT Fresnel kernel multiplies the input by `exp(i·k·r₁²/2z)` sampled at `dx`. That chirp's local frequency at the grid edge is `(N·dx/2)/(λz)`, so the sum is a valid quadrature only for `z ≥ N·dx²/λ`. Nothing in the function (or anywhere in the library — I grepped every `.py` for the formula) checks it. The only nod is a docstring sentence, `fresnel.py:228`: *"For very short distances (large Fresnel number), use ASM instead."*

**Measured** (`p11_fresnel_alias.py`) — N=128, dx=2 µm, λ=633 nm ⇒ `z_crit = N dx²/λ = 0.8088 mm`; input is a smooth Gaussian (w₀ = 40 µm) so the *field* is fully sampled and the chirp is the only error source. Oracle = the same continuous Fresnel integral evaluated with 8× oversampling in each axis, at three output points:

| z / z_crit | dx_out | rel err at 3 output points | warnings emitted |
|---|---|---|---|
| 0.25 | 0.500 µm | **3.34e-1, 3.43e-1, 3.92e-2** | 0 |
| 0.50 | 1.000 µm | 1.29e-4, 1.91e-4, 2.90e-4 | 0 |
| 1.00 | 2.000 µm | 3.99e-6, 3.12e-6, 2.01e-6 | 0 |
| 2.00 | 4.000 µm | 1.24e-6, 5.45e-7, 2.02e-6 | 0 |
| 4.00 | 8.000 µm | 4.61e-7, 2.24e-6, 1.47e-5 | 0 |

(Comparing single-FFT Fresnel against `fresnel_propagate_mft` agrees to 3e-14 at *every* z — both evaluate the same aliased discrete sum, so the MFT sibling is **not** a valid oracle for this question. That is why the direct-quadrature oracle was needed.)

**Impact** — reachable from `propagate(method='fresnel')`, `propagate_through_system(..., method='fresnel')` and, directly relevant to this project, `apply_real_lens(wave_propagator='fresnel')`'s in-glass leg (`elements/_lens_real.py:1893`). For a 1 mm element at N=1024, dx=1 µm, λ_med = 1.31 µm/1.5 ≈ 0.87 µm: `z_crit = 1024·1e-12/0.87e-6 = 1.18 mm`, so `z/z_crit = 0.85` — borderline; at λ_med 0.42 µm (visible, high-index) `z_crit = 2.44 mm` and `z/z_crit = 0.41`, inside the measured-bad zone.

**Fix** — the library already computes this exact number: `dispatch.py:941` forms `Q = λ|z|/(N·dx²)` (= z/z_crit) and trips ASM→SAS at `Q > 1`. Fresnel's validity is the complementary `Q ≥ 1`. Emit a `RuntimeWarning` (matching the SAS `z_limit` warning at `sas.py:204` and `_warn_mft_output_window` at `mft.py:91`) when `z < max(Nx·dx², Ny·dy²)/wavelength`, naming ASM as the alternative.

---

### [P1] SAS at `complex64` loses the ASM−Fresnel correction to catastrophic cancellation; the error grows linearly with `k·z`
`lumenairy/propagators/sas.py:223–224, 256, 275–277`

```python
target_fdtype = np.float32 if target_cdtype == np.complex64 else np.float64
f_x = xp.fft.fftfreq(N_new, d=dx).astype(target_fdtype)
...
h_AS = xp.sqrt((1.0 + 0j) - cx**2 - cy**2)
h_Fr = 1.0 - 0.5 * (cx**2 + cy**2)
delta_H = W * xp.exp(1j * k * z * (h_AS - h_Fr))
```
With a complex64 input, `f_x`/`cx`/`cy` are float32 and NEP-50 keeps `h_AS − h_Fr` in complex64. That difference is a cancellation of two quantities near 1: `sqrt(1−u) − (1 − u/2) = −u²/(2(1+sqrt(1−u))²)`, of order `u²/8`. The *absolute* error of each operand is ~eps, so the difference carries ~eps of absolute error regardless of how small it is, and it is then multiplied by `k·z`.

**Measured** (`p4_sas_rs_resample.py`), N_new = 1024, dx = 1 µm, λ = 633 nm:
```
max |Δ(h_AS - h_Fr)| float32 vs float64 = 9.091e-08
  -> phase error k*z*Δ = 2.919e-03 rad at z = 3.24 mm
  -> phase error k*z*Δ = 9.024e-01 rad at z = 1 m
```
0.90 rad at z = 1 m — and long distance is precisely SAS's reason to exist (`sas.py:114–122`). This contradicts the library's own complex64 contract at `fft_infra.py:190–195`: *"the in-library kernel-phase and phase-screen mitigations keep ASM and lens accuracy at single-precision-FFT noise-floor levels rather than degrading further with phase magnitude."* Here it does degrade with phase magnitude, linearly.

**Fix** — build `f_x`, `cx`, `cy`, `h_AS`, `h_Fr` in float64 unconditionally (they are field-independent and 1-D-broadcast, so the memory cost is nil) and cast only the finished `delta_H` to `target_cdtype` — the "f64-carrier-then-cast" recipe used in `fresnel.py:274–284` and `mft.py`. Better still, use the cancellation-free closed form
```python
u = cx**2 + cy**2                       # float64
s = np.sqrt(np.maximum(1.0 - u, 0.0))
dh = -u*u / (2.0 * (1.0 + s)**2)        # == h_AS - h_Fr, no cancellation
```
which is also exact in float64 at large `u`.

**Not a defect (checked):** SAS's *physics* is correct. Against an exact Fresnel-MFT on SAS's own output grid: rel L2 = 5.1e-5 (N=512, z=3.24 mm, dx_out=2 µm) and 1.0e-4 (z=6.47 mm, dx_out=4 µm), with `P_sas = P_fresnel_mft = P_in` to 7 digits, and the result is independent of `pad` (scale vs exact Fresnel = 1.000000 for pad = 1, 2, 3, 4). My first comparison against `angular_spectrum_propagate_mft` gave an exact 0.25 factor — that turned out to be the ASM-MFT oracle aliasing (its own `_warn_mft_output_window` fired), not a SAS bug.

---

### [P2] The pyFFTW ping-pong double buffer is serialised by ONE lock, so its two slots can never run concurrently
`lumenairy/propagators/fft_infra.py:1150` and `:2063–2074, 2117–2128, 2161–2172, 2200–2211`

`_build_plan_entry` allocates `n_bufs` aligned workspaces and **one `pyfftw.FFTW` plan bound to each of its own buffers**, but a single `'lock': threading.Lock()` for the entry. `_fft2`/`_ifft2`/`_fft2_nd`/`_ifft2_nd` then hold that one lock across `np.copyto(buf, x)` + `plan()`. The slot index is already advanced under `_PYFFTW_PLAN_LOCK` (`:1301–1307`), so two threads always get *different* plans and *different* buffers — there is nothing for them to race on, yet they serialise.

**Measured** (`p8_perf2.py`) — I wrapped the lock returned by `_get_or_make_plan` with a concurrency counter and ran 4 threads × 6 `_fft2` calls on the same `(1024,1024)` complex128 shape:
```
n_bufs = 2; slot buffers distinct: True; LOCK OBJECT SHARED: True
MEASURED max simultaneous threads inside the pyFFTW critical section: 1
```
**Impact** — any thread-parallel consumer at one grid size (per-polarisation amplitudes, per-realisation ensembles, a `parallel_amp`-style fan-out) gets zero FFT concurrency. It is partly masked today because FFTW itself runs 8 threads internally (`FFTW_THREADS = 8`), but it is a hard ceiling whenever `set_fft_threads(1)` or a below-knee grid makes intra-transform threading unprofitable.

**Fix** — one lock per slot:
```python
return {..., 'locks': [threading.Lock() for _ in bufs], ...}
...
return (entry['plans'][slot], entry['bufs'][slot], entry['locks'][slot], _nb)
```
No other change; the `_PYFFTW_PLAN_LOCK`-protected slot advance already guarantees exclusivity per slot.

---

### [P2] `_get_asm_H_natural`'s chunk sizing wastes ~3 full grids of transient; the streamed sibling's own cap fixes it byte-identically
`lumenairy/propagators/asm.py:419–426`

```python
ram = get_ram_budget()
row_cost = 3 * Nx * 16
max_chunk = max(1, int(ram * 0.1 / row_cost))
chunk = min(Ny, max_chunk)
```
On a large box this resolves to the whole grid below N≈8192. The streamed sibling states exactly this problem at `asm.py:548–558` and caps the band at `_ASM_STREAM_BAND_ELEMS = 1<<22` elements instead — **but only for itself**, so the default path (and every `_H_CACHE` cold build, and the batch variant) still allocates the full-grid float64 kernel workspace.

**Measured** (`p9_tilt_chunk.py`, tracemalloc, N=2048 complex128, one grid = 67.1 MB):

| chunk | peak traced during the H build | in grids | H byte-identical to whole-grid? |
|---|---|---|---|
| whole grid (default) | 272.7 MB | **4.06** | — |
| 512 rows | 136.4 MB | 2.03 | **True** |
| 128 rows | 84.5 MB | **1.26** | **True** |

(The full cold `angular_spectrum_propagate` call peaks at 438.7 MB = 6.54 grids; the streamed path at 339.8 MB = 5.06 grids.)

**Fix** — one line, provably bit-free (the code's own argument at `asm.py:548–550` — H is elementwise in (row, column) — is confirmed by the measurement above):
```python
chunk = max(1, min(Ny, max_chunk, _ASM_STREAM_BAND_ELEMS // max(Nx, 1)))
```
At N = 32768 this takes the kernel workspace from ~8.6 GB to ~134 MB.

---

### [P2] `resample_field`'s spline MTF is ~30× worse than documented, and it sits on the `fresnel`/`sas` legs of `apply_real_lens` and `propagate_through_system`
`lumenairy/propagators/mft.py:485–617`; docstring claim at `:531`

> *"Interpolation introduces a small error proportional to (dx_out/feature_size)^order. For order=3 (cubic), this is < 0.1% when features are sampled at >= 4 pixels."*

The implementation interpolates `E.real` and `E.imag` **separately** with `scipy.ndimage.map_coordinates(order=3)`. Cubic-spline interpolation of a near-Nyquist complex carrier attenuates it.

**Measured** (`p4_sas_rs_resample.py`), Gaussian envelope × a pure carrier, power ratio after resampling:

| carrier (cycles/pixel) | px per cycle | `P_out/P_in` (scale 0.5) | (scale 1.5) |
|---|---|---|---|
| 0.00 | ∞ | 0.999998 | 0.999997 |
| 0.05 | 20 | 0.999972 | 0.999970 |
| 0.10 | 10 | 0.999549 | 0.999547 |
| 0.20 | 5 | 0.990621 | 0.990619 |
| 0.30 | 3.3 | 0.931504 | 0.931502 |
| 0.40 | 2.5 | 0.718458 | 0.718456 |

The docstring's own case — "≥4 pixels per feature", i.e. 0.25 cycles/pixel — sits between the 0.20 and 0.30 rows, so the loss there is **between 0.9 % and 6.8 %**, versus a documented "<0.1 %". (I did not get a direct 0.25 measurement: the box was saturated by the concurrent audit fleet and the run was killed at the import step. The bracket from the two measured neighbours is enough to establish the claim is wrong by at least an order of magnitude.) `dx_out == dx_in` is an exact identity (rel L2 = 2.5e-16), so the error is purely the resampling MTF.

Why it bites in practice: the single-FFT Fresnel output grid is defined by `dx_out = λz/(N dx)`, so the residual output-plane chirp's local frequency at the grid edge `r = N·dx_out/2` is `r/(λz) = 1/(2 dx_out)` — **exactly Nyquist, at every z, by construction**. Any field with amplitude out at the rim is attenuated by the resample-back step.

Also: there is no anti-alias low-pass on downsampling — the docstring says "consider anti-alias filtering first" but nothing does it, so `dx_out > dx_in` folds high frequencies back in.

**Fix** — state the true MTF, and offer a band-limited (Fourier / chirp-Z) resampler for pitch changes of a sampled band-limited field; `_bluestein_centred_2d` in the same package already performs exactly this operation exactly, and `angular_spectrum_propagate_mft(z=0)` is a drop-in alias-free resampler for the square case.

---

### [P2] `system.py`'s Fresnel/SAS legs silently CROP the field when resampling back to the input pitch
`lumenairy/propagators/system.py:752–785`

After `fresnel_propagate` / `scalable_angular_spectrum_propagate` the field lives at `dx_new` over an extent `N·dx_new`. The chain then does
```python
E, _ = resample_field(E, dx_new, current_dx, N_out=E_in.shape[-1])
```
i.e. it re-samples onto `N·current_dx`. When `dx_new > current_dx` — the common diverging-beam case — everything outside the central `N·current_dx` is discarded (`map_coordinates(..., mode='constant', cval=0.0)`), with no warning.

**Measured** (`p6.py`), grid-filling top-hat (radius 0.42·N·dx), N=512, dx=2 µm, λ=633 nm, z=5 mm, `dx_new/dx = 1.5454`:
```
method='asm'     : P_out/P_in = 0.998990   relL2 vs ASM(bl=False) = 3.18e-02   (bandlimit, expected)
method='fresnel' : P_out/P_in = 0.996685   relL2 = 6.39e-02
method='sas'     : P_out/P_in = 0.950689   relL2 = 2.22e-01
```
Decomposition of the Fresnel case: the Fresnel step itself conserves power to 1.000000; the retained window before resampling holds 0.996980 of it — so essentially all of the loss is the crop.

**Fix** — warn when `dx_new > current_dx` (state the retained physical window), or thread the kernel's own pitch forward as `current_dx` (the element handlers already take `current_dx`/`current_dy`) instead of forcing a resample.

---

### [P2] `_PYFFTW_BAD_SHAPES` is keyed on the bare shape — one failure blacklists every dtype and both directions
`lumenairy/propagators/fft_infra.py:174` and `:1961–1970`

`_handle_pyfftw_failure` records `tuple(x.shape)` only. `_fft2`, `_ifft2`, `_fft2_nd`, `_ifft2_nd` all gate on `shape not in _PYFFTW_BAD_SHAPES`.

**Measured** (`p10.py`): after one simulated complex128 `MemoryError` at `(512,512)`, the blacklist is `{(512, 512)}` and a subsequent **complex64** transform at the same shape — half the memory, likely to succeed — also skips pyFFTW, as does the inverse direction. `reset_fft_backend()` clears it (verified), but nothing calls that automatically.

**Fix** — key on `(shape, dtype.str, direction)`, matching the plan-cache key.

---

### [P2] SAS is the only glass propagator that never receives `dy`
`lumenairy/elements/_lens_real.py:1886–1888` calling `lumenairy/propagators/sas.py:36–45`

`_propagate_through_glass(E, thickness, wavelength, n_medium_r, n_medium_kappa, dx, dy, ...)` forwards `dy` to the `fresnel`, `rs` and `asm` branches, but the `sas` branch calls `scalable_angular_spectrum_propagate(E, thickness, lam_medium, dx)` — a function whose signature has **no `dy` at all**. SAS does reject non-square *arrays* (`sas.py:174`) but nothing rejects a square array on an anamorphic *pitch*, so `dy ≠ dx` is silently propagated as if `dy == dx`. `system.py` guards the same branch properly with `_require_square_pitch` (`:768`); `_lens_real.py` does not.

**Fix** — add the same `_require_square_pitch`-style guard in `_propagate_through_glass`'s `sas` branch, or give SAS a `dy` (its kernel is separable in `x`/`y` apart from the `N_new` square assumption).

---

### [P3] `_asm_H_from_kz`'s complex64 "mitigation" is not more accurate than the cast it replaced, and is worse at large `k·z` — the docstring claims the opposite
`lumenairy/propagators/asm.py:99–112` (claim) and `:138–148` (code)

The docstring says the pre-v5.24.5 builders "cast the complex128 `exp` result straight to complex64 and carried ~1 float32-ULP of avoidable phase error per bin vs the correctly-rounded value". The naive cast **is** the correctly-rounded complex64 value: `np.exp(1j·x)` in complex128 argument-reduces correctly, and `astype(complex64)` then rounds each component to nearest. The mitigation instead computes `np.mod(kz*z, 2π)` — introducing `ulp(kz·z)` of *argument* error — then float64 `cos`/`sin`, then the same float32 rounding.

**Measured** (`p8`-style inline probe, N=4096, dx=1 µm, λ=1 µm, max |phase error| vs the complex128 reference over the propagating set):

| z | mitigated c64 | naive `astype(c64)` | ratio |
|---|---|---|---|
| 1e-4 m | 3.9684e-08 | 3.9684e-08 | 1.00 |
| 1e-2 m | 4.0117e-08 | 4.0117e-08 | 1.00 |
| 1 m (k·z ≈ 6.28e6) | 4.0787e-08 | 4.0787e-08 | 1.00 |
| 100 m (k·z ≈ 6.28e8) | **6.4146e-08** | 4.0674e-08 | **1.58 (worse)** |

The mitigation *is* genuinely needed on the JAX-x32 path, where `kz` itself would be float32 — which is what the same comment block's S2-3 note about `jnp.arange` correctly describes. The claim that it fixes the NumPy square/tilted/MFT builders is not supported by measurement.

**Fix** — correct the comment (and keep the code: it uses *less* transient memory than the complex128-exp route, ~40 vs ~56 bytes/element, so there is no reason to revert it — just stop claiming an accuracy win it does not deliver).

---

### [P3] `_build_asm_H_square`'s "bit-exact to the inline path" contract is false for odd N
`lumenairy/propagators/asm.py:230–233` (claim), `:261` vs `fft_infra.py:1630–1636`

`_build_asm_H_square` builds `fx = (arange(N) - N//2) / (N*dx)` (division); `_get_or_make_freq_grids` builds `fx = (arange(N) - N//2) * (1.0/(N*dx))` (multiply by reciprocal). These differ by up to 1 ULP whenever `1/(N·dx)` is not exactly representable.

**Measured** (`p6.py`), `_build_asm_H_square(N,dx,z,λ)` vs `fftshift(_get_asm_H_natural(...))`:
```
N=64,  dx=1.0 um, bl=T/F : byte-identical = True   (1/(N dx) = 15625.0 exactly)
N=256, dx=0.5 um, bl=T/F : byte-identical = True   (1/(N dx) = 7812.5 exactly)
N=255, dx=0.5 um, bl=True  : byte-identical = True
N=255, dx=0.5 um, bl=False : byte-identical = FALSE,  max|ΔH| = 9.096e-13
```
Physically ~1e-12 rad, i.e. irrelevant — but "bit-exact" is a pinned-bits contract and `_build_asm_H_square` is the `shack_hartmann` per-lenslet path.

Related, latent: `_get_or_make_bandlimit` (`fft_infra.py:1725–1726`) uses the **divided** form to label frequency bins, while the H it masks is built from the **multiplied** form — a 1-ULP label mismatch between mask and kernel. I could not make it flip a mask bin in 400 randomised `(N, dx, λ, z)` trials (0/400 differences), so it is latent only; but the two should use one expression.

---

### [P3] `angular_spectrum_propagate`'s documented memory is 2.2× low, and the H-build share is 63 % not "30–50 %"
`lumenairy/propagators/asm.py:674` ("Memory: approximately 3x the size of the input array (E_in, E_fft, H)"); `asm.py:329` / `fft_infra.py:1352–1353` ("~30-50% of total ASM time on 2k+ grids")

**Measured** (`p8_perf2.py`), N = 2048 complex128, one grid = 67.1 MB:
```
cold angular_spectrum_propagate    : traced peak 438.7 MB = 6.54 grids
same with stream_transfer_function : traced peak 339.8 MB = 5.06 grids
```
and, with pyFFTW plans already warm:
```
H build (cold)               1517.1 ms
fft2 + multiply + ifft2       888.7 ms   ->  H build = 63% of a cold ASM call
```
The 6.54-grid peak is dominated by the un-capped kernel workspace (see the chunk finding above); with a 128-row cap the same call would peak near 3.5 grids, which is much closer to the documented figure.

---

### [P3] `rs.py` claims RS and ASM "agree to machine precision"; they converge to ~2.8e-2
`lumenairy/propagators/rs.py:131–133`

> *"At large distances (z >> a²/λ), RS and ASM give identical results. For intermediate distances they agree to machine precision when ASM uses no band-limiting (`bandlimit=False`)."*

**Measured** (`p4_sas_rs_resample.py`), circular aperture a = 20 µm, λ = 633 nm, z = 100 µm, `bandlimit=False`, analytic on-axis `I = 4 sin²(k/2(√(z²+a²) − z)) = 0.61794`:

| N | dx | I_RS(0) | I_ASM(0) | rel L2 RS vs ASM |
|---|---|---|---|---|
| 512 | 500 nm | 0.59553 | 0.59842 | 5.22e-2 |
| 1024 | 250 nm | 0.57155 | 0.55523 | 3.35e-2 |
| 2048 | 125 nm | 0.59132 | 0.57897 | 2.82e-2 |
| 4096 | 62.5 nm | 0.60271 | 0.59080 | **2.76e-2** |

The difference plateaus at ~2.8e-2 rather than converging to zero. That is expected — RS point-samples the *spatial* Green's function and DFTs it on a 2N grid (so the discrete kernel is the aliased transfer function), whereas ASM samples the analytic transfer function on the N grid. Both bracket the analytic on-axis value and both converge toward it as `dx→0`. The **kernel signs and normalisations are correct** (see "checked and found correct"); only the docstring claim is wrong.

---

### [P3] `propagate()` has no `dy` — the dispatcher is square-pixel-only while every kernel under it is not
`lumenairy/propagators/dispatch.py:110–133`, `:1190–1230`

Measured signature: `['E_in','z','wavelength','dx','prescription','method','accuracy','output_grid','output_dx','return_result','method_kwargs']`. `_dispatch_to_method` forwards only `dx`. Meanwhile `PropagationResult` exposes `.dy` (`dispatch.py:515`), and `angular_spectrum_propagate`, `fresnel_propagate`, `fresnel_tf_propagate`, `rayleigh_sommerfeld_propagate` all accept `dy`. A caller who passes `dy=` gets it swept into `**method_kwargs` and forwarded to some kernels (asm, fresnel, rs) but rejected by others (sas, fraunhofer takes it, gbd/hf/hfpi do not) — an inconsistent, undocumented surface.

---

### [P3] `fresnel_tf_propagate` force-promotes to complex128 and is silently NumPy-only
`lumenairy/propagators/fresnel.py:149, 156–159`

`np.ascontiguousarray(E_in, dtype=np.complex128)` makes a complex64 caller pay 2× the FFT memory and time for a result that is cast straight back to complex64 at `:159`. Separately the function has no backend dispatch at all (every other member of the family has one) and hard-codes `_get_or_make_freq_grids(..., True)` — `xp_is_numpy=True` — at `:149`, so a CuPy input would either crash in `ascontiguousarray` or silently take host frequency grids.

---

### [P3] The `_H_CACHE` entry is handed out writeable by reference to internal callers
`lumenairy/propagators/fft_infra.py:1736–1741`, `asm.py:469–471`

`_h_cache_lookup` returns the stored array with no copy. The public entry point copies (`asm.py:791–798`, verified: the returned `H` does not share memory with the cache), but `_get_asm_H_natural` hands the live object to `angular_spectrum_propagate`, `angular_spectrum_propagate_batch` and `shack_hartmann`. **Measured:** two successive `_get_asm_H_natural` calls at the same key return arrays for which `np.shares_memory(...) is True`. The comment at `asm.py:469–471` says "callers must not mutate it in place"; a one-line `H.setflags(write=False)` before `_h_cache_store` would turn that convention into an enforced invariant at zero cost.

---

## Performance opportunities

1. **Cap the H-build chunk** (P2 above). Measured 4.06 → 1.26 full grids of transient at N=2048, byte-identical. At N=32768/complex128 that is ~8.6 GB → ~134 MB. One line.
2. **Per-slot plan locks** (P2 above). Restores 2-way FFT concurrency at one shape; measured concurrency is currently exactly 1.
3. **Drop the redundant `where` in every `kz` build.** `asm.py:267` (`np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)` — *doubly* redundant), `asm.py:444`, `asm.py:565`, and the ASM-MFT builder in `mft.py`. `np.sqrt(np.maximum(x, 0))` already returns exactly `0` where `x ≤ 0` — **verified identical over 1e5 random values including negatives**. Saves one full-chunk float64 temporary + one pass per chunk, per H build. `prop` is still needed for the evanescent mask inside `_asm_H_from_kz`.
4. **The pyFFTW path earns its keep.** Measured forward 2-D complex128 (min of 5, loaded box): N=1024 — `_fft2` 8.7 ms vs `scipy.fft workers=-1` 29.2 ms vs `numpy.fft` 51.1 ms; N=2048 — 146.9 / 285.3 / 372.8 ms; N=4096 — 788 / 1197 / — ms. No change recommended.
5. **`stream_transfer_function` costs ~2× the warm plain path** (measured 1848 ms vs 915 ms at N=2048) for a 1.5-grid peak saving, exactly as the docstring says. With opportunity (1) applied the plain path's peak drops below what the streamed path achieves today, which weakens the case for the streamed path below the `_H_CACHE_MAX_BYTES_PER_ENTRY` threshold.
6. **`fresnel_tf_propagate` at complex64** (P3 above): 2× memory and ~1.6× time for no accuracy gain over an f64-carrier-then-cast H.
7. **`_bluestein_axis_1d` does not use the `_H_FFT_CACHE`.** `_bluestein.py:218` recomputes `_fft_1d(h)` on every call, while the 2-D route caches its kernel FFT (`:450–465`). The 1-D kernel is two length-`L` vectors (0.15 MB at design-121 shapes, per the module's own note) — the cheapest possible cache entry, and the `separable=True` route is the one the memory work pushed callers toward.

---

## Alternative algorithms / methods

1. **Band-Extended ASM (BEASM)** — Zhang, Zhou, Jiao, *"Band-extended angular spectrum method for accurate diffraction calculation in a wide propagation range"*, Opt. Lett. **45**(6), 1543 (2020); and the shifted variant **shift-BEASM**. The Matsushima hard mask this library applies (`fft_infra.py:1644–1733`) *discards* the out-of-band spectrum, which is why `bandlimit=True` loses energy (measured `P_out/P_in = 0.99999502` at z=10 mm, N=512, dx=1 µm vs 1.00000000 with `bandlimit=False`). BEASM instead **resamples the frequency grid non-uniformly** so the chirp stays sampled, keeping the content the mask throws away, at the cost of a non-uniform (NUFFT / chirp-Z) inverse. It buys back long-distance accuracy exactly in the `Q > 1` band where this dispatcher currently jumps to SAS.
2. **Adaptive-sampling ASM (AS-ASM)** — Zhang, Zhou, Jiao, Opt. Express **28**(26), 39916 (2020). Same idea with an adaptive output grid; it removes the "which of ASM / SAS / Fresnel" decision entirely for a single-shot propagation, which would simplify `_auto_select_method`'s three-band rule.
3. **The `Q` bands are complementary and Fresnel is never selected.** `dispatch.py:941` computes `Q = λ|z|/(N dx²)`; ASM is valid for `Q ≤ 1`, the single-FFT Fresnel for `Q ≥ 1` (my P1 measurement pins the crossover: rel err 3.3e-1 at Q=0.25, 1.3e-4 at Q=0.5, 4e-6 at Q=1). Yet `auto` routes `Q > 1` to SAS (3 FFTs on a `pad·N` grid ⇒ ~12× the work of a single `N` FFT) and never picks `fresnel`. For a paraxial field in the `Q > 1` band, plain Fresnel is valid and roughly an order of magnitude cheaper; SAS's advantage is only its ASM-vs-Fresnel correction term, which matters at high NA. A NA test (or a cheap `|∇φ|` probe on the input) would let `auto` take the cheap branch when it is legitimate.
4. **Chirp-Z instead of a resample for the Fresnel output-pitch problem** — the library already has `fresnel_propagate_mft` (`mft.py:620+`, via `_bluestein_centred_2d`), which evaluates the same Fresnel integral *directly on the caller's output grid*. `system.py:752–766` and `_lens_real.py:1892–1895` instead call `fresnel_propagate` and then spline-resample back, which is both lossier (P2 above) and no cheaper (Bluestein is 3 FFTs of length `next_fast_len(N_in+N_out-1)` vs 1 FFT + a `map_coordinates` pass). Routing those two call sites to `fresnel_propagate_mft(..., dx_out=current_dx, N_out=N)` would remove the crop, remove the spline MTF, and keep the declared grid.
5. **Real-to-complex FFTs.** Not applicable here: every propagator input is a complex field and no kernel has Hermitian symmetry in the field. The only Hermitian object is `H` itself (`H(-f) = H(f)` for plain ASM, since `kz` is even), so **half of `H` need never be built or stored** — building the `fy ≥ 0` half and mirroring would halve the H-cache footprint and the kernel workspace. This composes with opportunity (1) and does not apply to the tilted builder (where `H` is evaluated at `f + f₀` and is not even).
6. **Shen & Wang (2006)** — already cited at `rs.py:148–150` — prescribes *integrating* the RS kernel over each pixel rather than point-sampling it. That is the source of the ~2.8e-2 RS-vs-ASM floor measured above and would let the docstring's convergence claim become true.

---

## Code organization observations

- **Comment-to-code ratio.** `fft_infra.py` is 2414 lines for roughly 900 lines of executable code; `dispatch.py` spends ~300 of its 1682 lines on one docstring. Several comment blocks are longer than the functions they annotate (`asm.py:480–516`, `_bluestein.py:77–100`, `dispatch.py:860–940`) and narrate audit history rather than behaviour. The `_auto_select_method` docstring documents a "DOE → hfpi" branch at length *and then* explains for 25 lines that the branch was deleted.
- **Two "is this a CuPy array" implementations** — `fft_infra._is_cupy_array` and `backend.array.is_cupy_array`, functionally identical, in the same package.
- **Three separate cache-with-byte-budget implementations** with slightly different eviction semantics: `fft_infra._h_cache_store`, `_bluestein._h_fft_cache_store` (which deliberately stops at one entry — documented at `_bluestein.py:126–131`), and `_PYFFTW_PLAN_CACHE`'s inline trim. The `_bluestein` one explicitly says it is "bringing a sibling cache up to a standard the library already sets" — that is an argument for one shared bounded-LRU helper, not three.
- **`snapshot_fft_state` omits `FFTW_MIN_SIZE` and `_PYFFTW_AUTO_PROMOTE_THRESHOLD`.** Both are module globals that change dispatch behaviour; neither has a setter, so this is consistent with the stated "setter-backed globals" rule, but `FFTW_MIN_SIZE` in particular is the kind of thing a tuned parent would have edited. Captured set verified at 19 keys (`p6.py`).
- **`backend/scipy.py:137–145` `lstsq`** dispatches to three functions with three different return contracts (`scipy.linalg.lstsq` → 4-tuple; `jnp.linalg.lstsq` → 4-tuple with different semantics; `cupy.linalg.lstsq` requires `rcond`). Same class of latent divergence as the `jv` bug.
- **`backend/random.py`** docstring (`:47`) refers to `rs.key` as the JAX key accessor; no such property exists (the attribute is `_key`, and `_rng` is unset on the JAX branch, so `rs._rng` raises `AttributeError`). The class makes **no** cross-backend reproducibility claim, so the "same seed → same numbers across backends" question has no contract to violate.
- **`propagation.py`'s "Return-type contract" block (`:22–38`) is incomplete.** It enumerates ASM / tilted ASM / RS as bare-ndarray and Fresnel / Fraunhofer as 3-tuple, and calls the split "intentional and stable", but omits `scalable_angular_spectrum_propagate` (also a 3-tuple, `(E, dx_out, dx_out)`), `fresnel_tf_propagate` (bare ndarray, same-grid) and the whole MFT family (bare ndarray at a caller-chosen grid). A reader taking that block as exhaustive would mis-handle SAS — which is exactly the case `propagate()`'s `auto` selector routes to most often.
- **`PropagationResult.__array__` does not accept numpy 2.x's `copy=` keyword** (`result.py:183–188`). MEASURED on numpy 2.4.6: `np.asarray(result)` works; `np.array(result, copy=True)` emits `DeprecationWarning: __array__ implementation doesn't accept a copy keyword…`; `np.array(result, copy=False)` **raises `ValueError: Unable to avoid copy while creating an array as requested`**. One-line forward-fix: `def __array__(self, dtype=None, copy=None)`. (The 2-item `__iter__` alongside `__array__` is deliberate and extensively documented at `:61–78`; `__all__` integrity is clean — 89 names in `lumenairy.propagators` and 57 in `…propagation`, all resolvable.)

---

### [P2] `richards_wolf_focus`: `E_z` appears to carry the opposite sign to `E_x`/`E_y` (the azimuth used for the depolarisation term is the aperture point's, not the ray's)
`lumenairy/propagators/vector_diffraction.py:249, 347, 396`

The transform-direction question is **RESOLVED and the code is right** — see the measurement below. What that resolution then exposes is a relative-sign question on `E_z`.

**Step 1 — which coordinate is the `pupil` array in? MEASURED.** A pupil ramp `exp(+2πi·u·x_p)` at NA=0.2, λ=633 nm, f=4 mm, with `u·λ·f = +2.3737 µm` predicted:
```
tilted pupil exp(+2pi i u x_p), u*lam*f = +2.3737 um -> measured centroid x_f = +2.3600 um
```
(0.6 % residual = intensity centroid vs. peak on a finite window.) Independent ray-trace check: the field at the exit pupil is `exp(-i k sqrt(x_p²+y_p²+f²))·exp(2πi u x_p)`, so the ray at aperture point `x_p` leaves with transverse direction `-x_p/f + uλ` and, over the distance `f` to the focal plane, lands at `x_p + f(-x_p/f + uλ) = +u·λ·f` — independent of `x_p`, i.e. a cleanly shifted focus at **+uλf**. The code's `np.fft.fft2` (`:396`, kernel `exp(-i k (x_p/f) x_f)` ⇒ `s_x = -x_p/f`) paired with `defocus = exp(+i k z cosθ)` (`:358`, `s_z = +cosθ`) gives exactly that. **So `fft2` is correct, and the `pupil` array is indexed by the PHYSICAL exit-pupil aperture coordinate** (the one you would measure with a ruler across the lens), not by the projected ray direction.

**Step 2 — the consequence.** `phi_p = arctan2(Yp, Xp)` at `:249` is therefore the azimuth of the *aperture point*, which differs from the *ray-direction* azimuth by exactly π. The standard aplanatic result (Richards–Wolf 1959; Leutenegger et al. 2006) is written in ray-direction azimuth `φ_ray`:

```
E_x, E_y  ~  cos²φ, sin²φ, cosφ sinφ      -> INVARIANT under φ -> φ+π
E_z       =  -(p_x cosφ_ray + p_y sinφ_ray) sinθ   -> FLIPS SIGN under φ -> φ+π
```
With `φ_ray = φ_p + π`, the correct z-component in the code's own coordinate is `E_z = +(p_x cos φ_p + p_y sin φ_p) sinθ`, but `:347` has `Pz = P*(-px*cp*s - py*sp*s)`, i.e. the negative of that. `E_x`/`E_y` at `:343–346` are unaffected because they are even in the azimuth.

**Impact and confidence.** This is a DERIVED result, not a measured one — `|E_z|²` is blind to it, so `debye_wolf_psf` and every intensity test are blind to it, and I could not construct a quick independent oracle for the *relative* sign. The magnitude is certainly right: measured `E_z` energy fraction **0.00062 at NA=0.05** (correctly vanishing in the scalar limit) and **0.22880 at NA=0.90** (peak `|E_z|²/|E_x|²_max = 0.1209`) — textbook-order for a uniformly filled aplanatic lens in air. Only the sign relative to `E_x` is in question, and it matters for focal-field handedness/spin, spin–orbit coupling, optical-force calculations, and any coherent superposition that uses all three components.

**What would settle it:** compare `E_z(x_f)` against an independent Richards–Wolf implementation (or against the analytic on-axis-adjacent limit `E_z ∝ -i·(x_f/|x_f|)·…` for an x-polarised low-NA focus, where `E_z` is odd in `x_f` and its sign relative to `E_x` is fixed) at a single off-axis point. If confirmed, the fix is one character at `:347`; if the intent was instead the projected-direction convention, then `:396` needs `ifft2` (with the `N²` normalisation folded into the prefactor) and `:347` is already right — but that second option is excluded by the measurement in step 1.

**Separately (P3): the convention is undocumented.** `sin_theta = rho_p/f` and `phi_p = arctan2(Yp, Xp)` read like the projected-direction convention, while the transform direction proves the array is in physical-aperture coordinates. A caller who builds an asymmetric pupil (coma, an off-centre obstruction, a segmented aperture) in the other convention gets a point-inverted PSF with no diagnostic. The docstring (`:30–92`) should state which coordinate `pupil` is indexed by.

---

## Unverified suspicions

- **numexpr byte-identity.** `asm.py:43–44` and `:150–153` claim `numexpr.evaluate('where(prop, exp(1j*kz*z), 0j)')` is byte-identical to `np.exp` ("measured max |diff| = 0.0"). numexpr is not installed here, so I could not test it. What would confirm it: install numexpr, and check *both* with and without MKL/VML (`numexpr.use_vml`), because VML's default accuracy mode is the plausible failure mode for a claim this strong. Worth re-testing because the claim gates a default-on fast path for every grid ≥ 512².
- **Odd-N tilted ASM.** The comment at `asm.py:1198–1204` argues the spatial `-N/2` vs frequency `-N//2` anchor mismatch "cancels exactly between demodulation and remodulation". I verified the algebra (a constant global phase commutes through the linear propagation and cancels against `conj(carrier)`) and verified the tilted kernel is exact at even N (below), but did not run an odd-N tilted case. A one-line test at N=257 would close it.
- **CuPy paths.** `cupy` is not installed; every `xp is cp` branch in `asm.py`, `rs.py`, `sas.py`, `mft.py`, `fft_infra.py` and `backend/*` was desk-checked only. Note `_get_or_make_freq_grids`/`_get_or_make_bandlimit` return **uncached** freshly-built device arrays on that path (by design), so the CuPy H build pays the 1-D vector construction every call.
- **`_h_cache_store` self-eviction.** If a caller sets `h_max_total_bytes` below one entry's size, the loop pops the entry it just inserted (`fft_infra.py:1787–1793`); `_bluestein._h_fft_cache_store` guards against exactly this (`:150–151`) and `fft_infra` does not. Unreachable with the shipped caps (2 GB/entry inside 8 GB total), so I did not force it.
- **`_auto_select_method` routing into a warned SAS regime.** For N=512/dx=2 µm/λ=633 nm, `auto` selects `sas` for `z ∈ (3.235 mm, 4141 mm)` while SAS's own `z_limit` is 2086 mm — so `z ∈ (2086, 4141) mm` is auto-routed into a kernel that then emits its own `RuntimeWarning`. I confirmed the numbers but did not confirm the accuracy cost in that window.

---

## Checked and found correct

- **ASM transfer function.** `H = exp(i·z·sqrt(k² − kx² − ky²))`, medium handled by the caller passing `λ_med` (`_lens_real.py:1881` `lam_medium = wavelength / n_medium_r`, threaded to `k`, the evanescent cutoff and the bandlimit consistently). Against the analytic Gaussian at N=1024/dx=0.5 µm/w₀=8 µm: rel L2 = **2.43e-4 / 4.86e-4 / 1.46e-3** at z = 0.5/1/3 z_R. Tilted plane wave: phase error vs `exp(i k z cosθ)` = 0, −4.5e-4, +8.8e-4 rad at θ = 0°, 5°, 20° (residual is the off-lattice carrier, not the kernel).
- **Evanescent handling and negative z.** Evanescent bins are *zeroed*, not decayed, so `z < 0` cannot blow up: on a sub-λ grid (dx = 0.2 µm, λ = 1 µm) with white-noise input, `max|E_out|` = 1.74 at z = +5 µm and 1.56 at z = −5 µm, both finite, `P_out/P_in = 0.127457` **identically for both signs**. The information loss is irreversible (forward-then-backward round-trip rel L2 = 0.934 on that field) — correct and unavoidable given the zeroing choice.
- **Energy.** Parseval holds to 1e-8 for a non-evanescent Gaussian at z = 0.1/1/10 mm with `bandlimit=False`; with `bandlimit=True` only the z=10 mm case loses anything (0.99999502).
- **`fftfreq`/`fftshift` centering, both parities.** `(arange(N) − N//2)·df` is exactly `fftshift(fftfreq(N,dx))` for even and odd N; the 2-shift fold `fftshift(ifft2(fft2(ifftshift(E))·H_natural))` is algebraically exact. ASM-MFT reproduces plain ASM on the same grid to **2.70e-14 (N=256)** and **2.82e-14 (N=257)**. Fresnel single-FFT reproduces `fresnel_propagate_mft` to **5.04e-14 at N=257** and 2.7e-14 at N=256 — the odd-N half-sample correction in `_centred_dft_halfpixel_args` is right.
- **Anamorphic bandlimit.** With `dy = 2·dx` the nonzero pattern of the returned `H` matches `(|fx| < Lx/(2λ|z|)) & (|fy| < Ly/(2λ|z|)) & (kz² > 0)` exactly — `dx` and `dy` are used separately.
- **H cache key.** `(Ny, Nx, dy, dx, λ, z, bandlimit, dtype.str, 'ASM')` is complete. Verified no stale hit across: `bandlimit` toggle (rel L2 = 8.64e-1 on a broadband field, order-independent and byte-reproducible in both orders), sign of `z`, `set_default_complex_dtype` mid-run (complex128 input unaffected — correct, dtype follows input; real input correctly switches the output dtype). JAX and CuPy arrays are correctly kept out of the host cache.
- **`stream_transfer_function` byte identity.** Verified **byte-identical** to the plain path at N=128 and 512, complex128 and complex64, with matching output dtype in all four cases (the defensive `spec.astype(target_cdtype)` at `asm.py:767` is what makes the complex64 case hold; `_fft2` does preserve complex64, verified).
- **`z == 0` identity.** `_build_asm_H_square` and `_get_asm_H_natural` both return all-ones (evanescent bins included), so ASM(z=0) reproduces the input on sub-λ grids.
- **Tilted ASM is exact, not an approximation.** Carrier removal + shifted `kz` + remodulation reproduces plain ASM of the same tilted field to **2.0e-14, 2.1e-14, 2.1e-14** at θ = 0.5°, 5°, 15° (carrier at 1 %, 14 %, 41 % of Nyquist), with and without the bandlimit. The bandlimit is correctly applied on the *shifted* frequencies. The `|fx0| < 1e-15` no-tilt shortcut is continuous (rel diff 2.3e-15 across it).
- **Fresnel prefactor and grid.** `exp(ikz)/(iλz) · exp(ik(x₂²+y₂²)/2z) · FFT[E·exp(ikr₁²/2z)] · dx·dy` with `dx_out = λz/(N dx)` — sign, pitch and centering all verified against `fresnel_propagate_mft` (5e-14) and, in the paraxial limit, against ASM.
- **`fresnel_tf_propagate` is the exact paraxial ASM.** Against an independently written `H = exp(ikz)exp(−iπλz(fx²+fy²))` on `fftfreq` grids: rel L2 = **1.09e-12**. Against exact ASM in the paraxial regime (N=512, dx=2 µm, w₀=30 µm, z=2 mm): 6.19e-6 — the expected `O(θ⁴)` truncation, not a bug. `z=0` is the exact identity and `z<0` is accepted, as documented.
- **RS kernel sign and normalisation.** `h = (z/(2πr²))(1/r − ik)e^{ikr}` = `−(1/2π)∂_z(e^{ikr}/r)`, whose `kr ≫ 1` limit is `(1/(iλ))·cosθ·e^{ikr}/r` — the standard forward Huygens–Fresnel kernel under `exp(−iωt)`/`exp(+ikz)`. The `r = 0` case is non-singular because `r = √(x²+y²+z²) ≥ z > 0` and `z ≤ 0` hard-raises.
- **RS zero-padding is a true linear convolution.** I worked the index algebra: with the kernel on `2N` samples centred at index `N` and `ifftshift`-ed, and the input placed at `[N//2 : N//2+N]`, the extracted block `E_conv[N//2 : N//2+N]` equals `Σ_q E_in[q]·h((s−q)·dx)` — the exact linear convolution over the full needed lag range `−(N−1) … (N−1)`, for both parities, with output index labelling identical to the input's. No wrap-around contamination.
- **Bluestein / chirp-Z.** `_bluestein_2d` (2-D route **and** `separable=True` route) and `_bluestein_centred_2d` all agree with an explicitly written direct DFT matrix at N_in=64 → N_out=48 to **1.2e-14 – 1.4e-14**, for both `sign = ±1`. The `next_fast_len(N_in + N_out − 1)` padding and the folded-kernel circular indexing are correct. The `H_FFT` cache key `(αx, αy, Nx_in, Ny_in, N_out_x, N_out_y, sign, dtype)` is complete (`Lx`/`Ly` are functions of those); entries are `np.copy`-ed on both store and hit, which is required given `_fft2`'s buffer-ownership contract.
- **SAS kernel.** `pad`-independent, power-conserving, and correct against exact Fresnel — see the P1 SAS entry. The cache key `(N_new, dx, λ, z, skip_final_phase, dtype, 'SAS')` is complete (all three kernels depend only on `N_new·dx`, not on `pad` and `N` separately).
- **`resample_field` at `dx_out == dx_in`** is the exact identity (rel L2 = 2.47e-16), and its `N_out`/`order`/`dx` validators all behave as documented.
- **`propagate_ensemble` adds intensities, not fields.** `ensemble.py:406–430`: `I_acc += |E_k|²` then `/ n_real`. The accumulator dtype is derived from the input complex dtype (`complex64 → float32`), the empty-ensemble and shape-drift cases both hard-raise, and the CuPy/JAX/NumPy branches are separated correctly.
- **`combine_patch_fields` is a partition of unity by construction** (`subaperture.py:263–277`): it accumulates `Σ F_i·w_i` and `Σ w_i` and divides, so the weights sum to 1 wherever any patch has support, with a `> 1e-12` guard elsewhere. No window shape can break the normalisation.
- **4f relay end-to-end** (`propagate_through_system`, ASM, N=512, dx=2 µm, f=10 mm, off-axis Gaussian at +20 µm): output centroid **−9.999874 px** vs input **+10.000000 px** (exact inversion to 1.3e-4 px), power ratio 0.999777, and the phase over the spot (|E| > 0.2 max) is flat to **std 0.0140 rad, PTV 0.0585 rad**.
- **Two-lens telescope vs Gaussian ABCD** (f₁=10 mm, d=30 mm, f₂=20 mm, w_in=60 µm, λ=633 nm — deliberately *not* in the geometric limit, since z_R = 17.9 mm < d): measured output `2σ` = **156.6878 µm**; hand-computed ABCD `q`-parameter chain gives **156.69 µm**. Agreement to 5 significant figures; power ratio 1.000000. The naive geometric `f₂/f₁ = 2` would give 120 µm, so this is a genuine test of the full Gaussian propagation, not of a ratio.
- **`_validate_mft_output_grid` / `_warn_mft_output_window`** correctly reject `dx_out ≤ 0`, non-finite, and `N_out < 1`, and the periodicity warning fires with the right numbers (it correctly flagged my own deliberately over-wide ASM-MFT oracle, which is what exposed my false-positive SAS finding).
- **`backend/array.py`** uses `isinstance` throughout — no `hasattr(x, 'device')` duck-typing anywhere, so the numpy-2.x `.device` trap is genuinely closed (and `fft_infra._is_cupy_array:56–71` documents and fixes the same thing).
- **`backend/fft.py`** passes no `norm=` anywhere, so every backend uses the numpy default (unnormalised forward, `1/N` inverse) — consistent. No `rfft` usage, so no Hermitian-handling risk.
- **`_asm_apply_H_streamed`'s stated non-bit-identical edge case** (`asm.py:519–528`: `(inf+0j)·(1+0j)` at `z == 0`) is correctly analysed and correctly documented as unreachable from a finite input.
- **`richards_wolf_focus` reproduces the scalar Airy pattern at low NA.** MEASURED: with a uniform x-polarised pupil at NA=0.05, λ=633 nm, f=4 mm, Np=512 (array half-extent 2× the geometric rim so the rim mask genuinely bites), the focal intensity cut agrees with `[2 J₁(v)/v]²`, `v = k·NA·r`, to **rel L2 = 8.09e-04 over 6 Airy radii**. That single number validates the apodisation, the Jones weights, the prefactor and the focal pitch together. The `E_z` energy fraction correctly vanishes in that limit (**0.00062**) and rises to **0.22880 at NA=0.90** (peak `|E_z|²/|E_x|²_max = 0.1209`) — textbook-order for a uniformly filled aplanatic lens in air. *(My probe also printed a "first zero radius" of 79.1 µm against 0.61λ/NA = 7.72 µm; that is my zero-finder latching onto a later local minimum, not a library result — the 8.09e-04 Airy agreement over the whole 6-radius window is the meaningful number.)*
- **`richards_wolf_focus` — everything except the `E_z` relative sign.** The aplanatic apodisation is `1/sqrt(cos θ)` (`vector_diffraction.py:262–263`), which I re-derived independently and confirm is the right weight for a **Cartesian-pupil FFT** evaluation: `dx_p dy_p = f² sinθ cosθ dθ dφ` ⇒ `sinθ dθ dφ = dx_p dy_p/(f² cosθ)`, so `sqrt(cosθ)·(1/cosθ) = cos^(-1/2)θ`. The `E_x`/`E_y` Jones weights at `:343–346` are exactly the standard aplanatic `R_z(φ)R_y(θ)R_z(−φ)` result (Richards–Wolf 1959; Leutenegger et al. 2006) and are invariant under the `φ → φ+π` azimuth ambiguity discussed in the `E_z` finding above; the `E_z` **magnitude** at `:347` is also standard (only its sign relative to `E_x`/`E_y` is in question). The prefactor `(-i k/(2π f))·exp(+i k f)·dx_pupil²` at `:430–433` has the correct `1/f` amplitude (`1/f²` intensity) scaling and the `exp(+ikf)` sign that matches the library's `exp(-iωt)`/`exp(+ikz)` convention. The pupil is cast to the default complex dtype at `:111` *before* `Ex = np.zeros(..., dtype=pupil.dtype)` at `:353`, so a real-valued amplitude mask cannot silently lose the imaginary part of the focal field. The pad/crop registration at `:387/:393` correctly uses the floor anchors `N_focal//2 - Np//2` / `Np//2 - N_focal//2`, and there are three independent, correctly-gated diagnostics for the array-limited-NA, `N_focal < Np` crop, and `dx_focal` mismatch failure modes.
- **The Matsushima asymptote substitution** (`fft_infra.py:1652–1696`) is one-sided safe as claimed: `L/(2λ|z|) > 1/(λ√((2z/L)²+1))` always, and the worst *practically relevant* over-width is 1.41× at `z = L/2` (at smaller `z` the asymptote already exceeds the `1/λ` evanescent cutoff, so it filters nothing that the evanescent mask does not). The measured over-width table in that docstring reproduces.

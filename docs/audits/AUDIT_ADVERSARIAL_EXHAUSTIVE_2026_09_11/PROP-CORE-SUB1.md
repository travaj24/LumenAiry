# PROP-CORE-SUB1 — physics/numerics audit of 5 Lumenairy propagators (hf / hfpi / vectorial_hfpi / mhs / rs kernel / Richards–Wolf)

*(Delegated sub-audit of the PROP-CORE partition; the sub-auditor reported directly to the orchestrator and did not write a report file, so this file reproduces its report verbatim. Orchestrator note on finding #1: PROP-CORE measured the OPPOSITE conclusion with a pupil-ramp test (focus shift +u·λ·f); the two auditors used different pupil-indexing conventions (physical aperture coordinate vs ray direction). The code is self-consistent with the aperture-coordinate convention that every other Lumenairy propagator uses, so the orchestrator classifies #1 as a P3 documentation hazard — the convention must be stated — rather than a physics defect. All other findings stand as reported.)*

Scratch scripts: `…/scratchpad/PROP-CORE-SUB1/` (`t1`–`t16`). No repo file was created, modified or deleted.

Two P0s. Both are silent, both are on all-default paths, and one of them (the Richards-Wolf one) bites exactly the asymmetric-pupil case a metasurface project cares about.

---

## Findings

### 1. [P0 as reported — see orchestrator note] `vector_diffraction.py:396` — Debye-Wolf transverse kernel sign; the focal field is returned 180°-rotated about the optical axis under the ray-direction pupil convention

```python
396:  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
```

`np.fft.fft2` implements `Σ exp(−2πi·nm/N)`. With the centred-grid convention (`x_p=(j−Np/2)dx_p`, `x_f=(i−N/2)dx_f`) and `dx_focal = λf/(N_focal·dx_pupil)` (line 135), this evaluates `exp(−i·k(x_p x_f + y_p y_f)/f)`. The Richards-Wolf integral quoted in the module's own comment (lines 406-410) requires `exp(+i k·r)`. The **axial** factor at line 358 (`np.exp(1j*k*z*ct)`) does use `+`, so the returned volume is `E_true(−x, −y, +z)` — a 180° rotation about z, not a parity inversion.

**Measured** (`t3`, `t4`, `t11`):
- Off-centre sub-aperture at `x_c=+300 µm`, `f=2 mm`: focal phase ramp `d(arg Ex)/dx_f = −1.488286e6 rad/m` vs Debye-Wolf prediction `+1.488906e6 rad/m` → ratio **−0.999583**.
- Independent (θ,φ) Gauss-Legendre polar-quadrature Debye-Wolf oracle, Gaussian-apodised pupil + 0.35 waves Zernike coma, NA=0.3, f=2 mm. Full `(Ex,Ey,Ez)` vector relL2 vs oracle at `+P`: **0.74 … 9.9**. At `−P`: **1.9e-6 … 6.5e-5** (the quadrature floor). Same at `z=+2 µm` defocus (4.0e-6 at −P vs 1.11 at +P) — confirming only the transverse kernel flips.
- Blast radius, `‖PSF − rot180(PSF)‖/‖PSF‖`: uniform disc **1.1e-16**, defocus **1.6e-16**, 0.3 wv coma **0.418**, 0.3 wv x-tilt **1.285**. Invisible for any 180°-symmetric pupil; immediately wrong for coma/tilt/decentred/DOE/metasurface pupils, and for any coherent superposition.

**Fix (as proposed)**: line 396 → `np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded))) * (N_focal * N_focal)`. Verified: the two forms differ by exactly a both-axis reversal (2.5e-16 on a random array), and after the reversal every probe point matches the oracle to ≤6.5e-5 (`t5`).

*(Orchestrator: under the aperture-coordinate convention a ray from x_p = +300 µm arrives at the focus travelling in −x, so a NEGATIVE focal phase ramp is the physically correct answer; the "prediction" of + assumes the array is indexed by ray direction. Both conventions appear in the literature; the docstring must state which one `pupil` uses.)*

---

### 2. [P0] `rs.py:248-252` (reached by `hf.py:131`) — the point-sampled RS Green's function aliases on ordinary grids; up to **26× energy creation** with all-default arguments

`h` is point-sampled on the padded 2N×2N grid (lines 248-252). Its phase gradient is `k·sinθ·dx`, which exceeds the π/px Nyquist limit whenever `z < 2·N·dx²/λ`. Nothing checks this. This is the kernel behind `propagate_huygens_fresnel` (`hf.py:84-85` → `130-133`), `propagate(method='rs')` and `propagate(method='hf', z=…)`.

**Measured** (`t12`), Gaussian w0=6 µm, z=50 µm, λ=633 nm, vs an exact Hankel angular-spectrum oracle; ASM on the *identical* grid as control:

| N | dx | kernel rad/px | RS relL2 | **RS power/P_in** | ASM relL2 | ASM power/P_in |
|---|---|---|---|---|---|---|
| 64 | 0.5 µm | 2.68 | 2.9e-4 | 1.000 | 1.1e-3 | 1.000 |
| 64 | 1 µm | 7.82 | 1.5e-1 | 1.0225 | 6.1e-8 | 1.000 |
| 64 | 2 µm | 18.5 | 4.50 | **21.44** | 5.3e-8 | 1.000 |
| 128 | 1 µm | 9.25 | 2.08 | **5.31** | 6.1e-8 | 1.000 |
| 128 | 2 µm | 19.5 | 4.95 | **25.70** | 5.3e-8 | 1.000 |

z-scan at N=128/dx=1 µm/w0=10 µm (`t7`): relL2 = **2.85** (z=25 µm), **2.04** (50 µm), **4.8e-2** (100 µm), then 3.8e-8 from 200 µm onward.

**Fix (verified, `t14`)**: build `H` in the frequency domain on the same padded grid — the RS-I transfer function is exactly `exp(i k z √(1−(λfx)²−(λfy)²))` with the evanescent cut — instead of FFT-ing a point-sampled `h`. Measured relL2 **6.08e-8 / 5.26e-8** with power exactly **1.000000** on every failing grid. At minimum, raise when `z < 2*N*dx**2/wavelength`.
*Do not* use a spatial hard mask at `ρ_c = z·s_c/√(1−s_c²)`: I tried it and it leaves 0.41/0.52 residual error and power 1.16/0.54 (`t13`).

---

### 3. [P1] `rs.py:306-321` — `bandlimit=True` is all-pass exactly in the aliasing regime, and harmful outside it

The cutoff `fx_max = Lx2/(2λ|z|)` (line 311) exceeds the grid Nyquist `1/(2dx)` precisely when `z < 2·N·dx²/λ` — **algebraically the same condition** under which the spatial kernel aliases. Root cause: Matsushima's criterion is derived for the ASM *transfer function*; masking `H = FFT(h)` after `h` has already been aliased cannot undo the aliasing.

**Measured** (`t7`), N=128/dx=1 µm: at z = 25/50/100/200/300/400 µm the mask is all-pass and `bandlimit=True` output is `np.array_equal`-identical to `bandlimit=False`, while both are 2.85/2.04/4.8e-2/… wrong. At z=3 mm the mask does bite and makes it **5 decades worse**: relL2 3.64e-8 (False) → **1.03e-3** (True).

**Fix**: as #2. Document that `bandlimit` on RS is not a near-field remedy.

---

### 4. [P1] `hf.py:179-183` and `mhs.py:633-637` — resample + Parseval renormalisation **fabricates energy** when the target window is smaller than the source's

Both sites rescale by `sqrt(p_in/p_out)` on the stated grounds that "bicubic `map_coordinates` introduces a small power drift". That conflates interpolation error with genuine physical cropping, then hides the crop.

**Measured** (`t8`): `propagate_huygens_fresnel_freespace(E, 1e-3, 633e-9, 2e-6, output_dx=0.5e-6)` on N=64 — the requested ±16 µm window genuinely contains **67.27 %** of the native-grid power; the returned array carries **100.00 %**. Amplitudes inflated **1.219×** (48.6 % in intensity). Direct `resample_field` probe: `dx_out=0.5dx` → true 99.62 %, renorm 1.0014×; `dx_out=0.25dx` → true **75.17 %**, renorm **1.1232×**.

**Fix**: compute the truth power inside the target window on the *source* grid and rescale by `sqrt(p_window/p_out)`, or drop the renorm and warn when `N_out*dx_out < N_in*dx_in`. The block is duplicated verbatim; `hf.py`'s cross-references to it are stale (`hf.py:113` says `mhs.py:573-611`, `:159` says `583-587`, `:176` says `602-606`; actual = 584-642 / 614-618 / 633-637).

---

### 5. [P1] `hfpi.py:295` and `vectorial_hfpi.py:207` — `wavelength: float = 0.0` default silently drops the `1/(iλ)` Kirchhoff prefactor

`inv_i_lambda = (1.0/(1j*wavelength)) if wavelength > 0 else 1.0` (`hfpi.py:354`, `vectorial_hfpi.py:270`). Both functions are in `__all__`.

**Measured** (`t9`): omitting `wavelength` from `apply_aperture_diffraction` changes every path weight by **exactly 1/λ = 1.5798e6** in magnitude *and* **−90°** in phase, with **zero warnings**.

**Fix**: make `wavelength` keyword-only with no default, or raise when `wavelength <= 0`.

---

### 6. [P1] HFPI's estimator is not the Huygens-Fresnel integral — now quantified exactly

`hfpi.py:563-600` admits qualitatively that the per-path `1/r` and the output-binning Jacobian are missing. I derived and **confirmed the exact bias law**:

> `E[HFPI(p)] / E_true(p) = dx_out² · cosθ / (N_src_px · r)`

**Measured** (`t10`), Gaussian w0=12 µm, N=64, dx=2 µm, **24 M paths, cone 0.05 rad, occupancy 1.000** (so this is not shot noise):

| z | `|HFPI|/|ASM|` | predicted `dx²/(N_src_px·z)` | meas/pred |
|---|---|---|---|
| 2 mm | 4.91609e-13 | 4.88281e-13 | **1.0068** |
| 4 mm | 2.42921e-13 | 2.44141e-13 | **0.9950** |

ratio(2 mm)/ratio(4 mm) = **2.024** — the 1/r signature (1.00 would mean correct).

The consequence is stronger than "not photometric": the returned amplitude depends on the **output pixel area** and the **source pixel count**, so merely rebinning the grids changes the answer with no physics change. Also measured (`t9`), default cone, 64×64 grid: shape relL2 vs ASM after best global rescale = **0.9986** at 8 M paths, occupancy 0.277, and `|E|max` moves 14× between 2 M and 8 M paths (no convergence).

Severity P1 not P0 **only because** `propagate_hfpi` warns. But the warning lives solely on that thin wrapper: `propagate_hfpi_freespace_aperture` (`hfpi.py:627-637`), `propagate_hfpi_through_prescription` (`hfpi.py:852-914`) and **all of `vectorial_hfpi.py`** carry no such warning and are all exported.

**Fix**: at bin time multiply each path weight by `r_path · N_src_px / (dx_out² · cosθ_out)` — `r_path` is already available as `paths.opl` (reset at each re-emission, so it is exactly the last-leg length) and `cosθ_out` as `paths.directions[...,2]`. Also importance-sample source pixels by `|E_in|²` instead of uniformly (`hfpi.py:193-194`).

---

### 7. [P2] The under-sampling guard's own prescribed remedy is unreachable from every end-to-end entry point

`accumulate_to_grid`'s message (`hfpi.py:462-466`) says to "narrow `cone_half_angle` from its ~90-degree default".

**Measured** (`t9`): `propagate_hfpi(..., cone_half_angle=0.05)` → `TypeError: propagate_hfpi_freespace_aperture() got an unexpected keyword argument 'cone_half_angle'`. Identical for `propagate_hfpi_freespace_aperture` (`hfpi.py:609-626`) and `propagate_vector_hfpi_freespace_aperture` (`vectorial_hfpi.py:393-410`). Only `propagate_hfpi_through_prescription` (`hfpi.py:849`) exposes it.

**Fix**: add `cone_half_angle` to both free-space signatures and forward it to `init_paths_*` / `apply_*_aperture_diffraction`.

---

### 8. [P2] `vectorial_hfpi.py:298-390, 393-464` — no sampling-adequacy guard and no `on_undersampled` knob at all

**Measured** (`t9`): 20 000 paths on a 64×64 grid → **2 of 4096 pixels non-zero (0.05 %), zero warnings**. The scalar twin on identical settings raises a `RuntimeWarning`. The v5.31 guard (`hfpi.py:374-387, 442-475`) was never mirrored.

---

### 9. [P2] `hfpi.py:747-748` — `init_paths_stratified` allocates far **more** paths than requested when strata counts are supplied

`n_per = max(1, n_paths // n_total)`; `n_paths_actual = n_per * n_total`. When `n_total > n_paths`, `n_per` clamps to 1 and `n_paths_actual = n_total`.

**Measured** (`t9`): `n_paths=1000, n_strata_xy=(32,32), n_strata_dir=(32,32)` → `len(bundle) = 1,048,576` (**1049× requested**, 25 MB for `positions` alone). `(8,8)/(8,8)` with `n_paths=100` → 4096 (41×). At `(64,64)/(64,64)` this is 16.7 M paths from a 1000-path request (`np.indices` alone = 537 MB).

The in-code comment at `hfpi.py:744-746` ("If user requested fewer than n_total, sample only n_paths strata uniformly without replacement") describes behaviour that does not exist; the `[:n_paths_actual]` truncation at 769-772 is always a no-op.

**Fix**: implement the documented sub-sampling (`rng.choice(n_total, n_paths, replace=False)`), or raise when `n_total > n_paths`.

---

### 10. [P2] `vector_diffraction.py:111, 263, 265, 353-355, 434` — the single-precision path costs precision and saves nothing

The comment at lines 350-352 claims "allocate output in the input pupil's dtype so the `precision='single'` path stays complex64 end-to-end".

**Measured** (`t11`, dtype chain confirmed directly): with `set_default_complex_dtype(np.complex64)` the output dtype is **complex128**. Mechanism, confirmed element by element on numpy 2.4.6:
- `P = pupil(complex64) * apod(float64) * in_pupil(bool)` → **complex128** (line 265; `apod` is float64 at line 263), so `Px/Py/Pz` and every FFT run in double regardless;
- lines 398-400 then write the complex128 FFT result into the **complex64** `Ex/Ey/Ez` buffers allocated at 353-355 — a pure round-trip truncation;
- line 434 multiplies by the **complex128** `rw_prefactor`, promoting straight back.

Measured cost: relL2(single-config, double-config) = **1.755e-08**, for zero memory or time benefit.

**Fix**: cast `apod`, `in_pupil`, `c`, `s`, `cp`, `sp` to the pupil's real dtype and build `rw_prefactor` in the matching complex dtype — or drop the complex64 buffers and admit the function is double-only.

---

### 11. [P2] `mhs.py:162-165` — `MhsPipeline._validate` ignores `HuygensSurface.centre`

**Measured** (`t11`): a chain whose subdomain 0 ends at `centre=(0,0)` and subdomain 1 starts at `centre=(50e-6, 0)` is **accepted** and runs to completion; the 50 µm transverse jump is silently discarded (each propagator works on its own `in_surface`). The z/Ny/Nx/dx mismatch *is* caught (control passes).

**Fix**: add `cur.out_surface.centre != nxt.in_surface.centre` to the test at lines 162-165.

---

### 12. [P2] `hf.py:345-393` — `propagate_huygens_fresnel_with_opl_callable` is O(N_in²·N_out²) in a pure-Python per-output-pixel loop

17 full-input-grid `opl_fn` evaluations per output pixel (1 + 16 for the Van Vleck stencils at 352-375), with `chunk_output` deprecated to a no-op (`hf.py:279-287, 301-307`) so there is no chunking whatever.

**Measured** (`t16`, after warm-up): cost/(N_in²·N_out²) ≈ **3.7e-7 s** (3.682e-7 at 48→16, 3.747e-7 at 64→16). Extrapolating: 128²→128² ≈ **100 s**; 512²→512² ≈ **7.2 h**.

**Fix**: broadcast `s2x/s2y` as a third axis and evaluate in output-pixel chunks (resurrect `chunk_output` with real meaning).

---

### 13. [P2] `hfpi.py:1086-1093` — the prescription walk force-copies the whole bundle to host NumPy

`np.asarray(... to_numpy(...) ...)` for positions/directions/opl/alive before every `trace` call, so CuPy inputs lose device residency each segment and JAX tracing cannot survive it. The module docstring (`hfpi.py:21-27`) advertises "The full pipeline is written against `array_namespace`, accepting NumPy / CuPy / JAX source fields and returning the same backend."

---

### 14. [P3] `hf.py:1-16, 72-85, 99-100` — the module docstring and the advertised entry point contradict the code

The module docstring says it "Implements the direct Huygens-Fresnel diffraction integral with the **Van Vleck density correction** in the integrand", and `propagate_huygens_fresnel` is "the recommended entry point for new code". That entry point contains **no Van Vleck factor and no HF quadrature** — it is a 3-line delegation to `rayleigh_sommerfeld_propagate` (lines 130-133). Its own docstring (99-100) says "with the standard `1/(i lambda z)` Van Vleck factor"; the kernel actually applied (`rs.py:252`) is `(z/(2πr²))(1/r − ik)exp(ikr)`, whose leading term is `cosθ/(iλr)`, not `1/(iλz)`.

---

### 15. [P3] `rs.py:118-134` — docstring claims measured false

- "Near-field propagation (z ~ a few wavelengths) where ASM's band-limiting can suppress valid high-frequency content" — measured the **opposite**: at z=25/50/100 µm, ASM with its default `bandlimit=True` is exact to 3.7e-8 while RS is 285 % / 204 % / 4.8 % wrong.
- "For intermediate distances they agree to machine precision when ASM uses no band-limiting (`bandlimit=False`)" — measured relL2(RS, ASM bl=False) = **4.8e-2** at 100 µm, **2.04** at 50 µm, **2.85** at 25 µm.

---

### 16. [P3] Docstring/code contradictions in the HFPI family

- `hfpi.py:186-188` says the weight is `E_in[i,j]*cos(theta)*dx**2`; the code (222-225) also applies `complex(1/(iλ)) * 2π(1−cosθ_max)/n_paths`.
- `vectorial_hfpi.py:4-8` and `217-218` claim an "**m-theory dipole obliquity tensor** for vector-correct secondary-source amplitudes". There is no tensor: line 268 applies the same scalar `0.5*(cosθ_in + cosθ_out)` as the scalar module, and `propagate_vector_to_plane`'s own docstring (172-174) concedes propagation-induced polarization rotation is neglected. The only vector operation is the **default-off** transverse projection at 278-286, which additionally drops the longitudinal component and so is not energy-conserving.

---

### 17. [P3] `hfpi.py:109-118, 121-126` — `_spawn_rng` reproducibility and a silent swallow

- `except Exception: pass` around the JAX `fold_in` branch falls through to "return as-is", giving both streams the **identical key** — the exact correlation the function exists to prevent — with no diagnostic.
- The `np.random.Generator` branch calls `rng.spawn(...)`, which **mutates the caller's generator**; two `_spawn_rng(rng, 0)` calls on the same object give different streams, so results are not reproducible across repeated calls. Only the `int` branch (105-108) is deterministic.
- Minor: `hf.py:552-557` falls back from `beam_d4sigma` silently (the sibling `decompose_lg` fallback at 599-615 correctly warns).

---

### 18. [P3] `mhs.py` — what it actually is

**`mhs.py` implements no propagation physics of its own.** The module docstring (lines 9-11) says "At each Huygens surface, the ray bundle is converted to a complex field via a Huygens-surface integral, and that field is the new source for the next subdomain." There is no ray bundle, no Huygens-surface integral and no ray tracing anywhere in the file. What exists — and what lines 33-37 then correctly admit — is a **structural framework**: `MhsPipeline.run` (183-282) is a `for` loop calling `sub.propagator(E, in_s, out_s, **kwargs)` and collecting results. All physics is delegated: ASM (390-396), a hard aperture mask (431-444), GBD (467-481), the dispatcher (584-682).

The only physics literally in the file is the hard mask and the maslov-branch resample + power renorm (621-641) — and that renorm is **finding #4**, duplicated verbatim in `hf.py:171-187`. Secondary duplication: `aperture_subdomain._prop` (437-439) re-implements `HuygensSurface.grid()` (77-88) inline instead of calling it. Also `run(..., return_result=True)` stamps `wavelength=0.0` by default (`mhs.py:192, 269-279`) even though every subdomain already carries `wavelength` in its `kwargs` — measured `run(E, return_result=True).wavelength == 0.0`.

---

## Checked and found correct

1. **Richards-Wolf apodization** (`vector_diffraction.py:262-263`). `1/sqrt(cosθ)` is **right given the Cartesian FFT measure**: `dx_p dy_p = f² sinθ cosθ dθ dφ`, so `sqrt(cosθ)` (aplanatic/Abbe-sine) ÷ `cosθ` (Jacobian) = `1/sqrt(cosθ)`. The derivation in the comment at 251-261 is correct. *Not* a missing-apodization bug.
2. **Polarization rotation matrix** (`vector_diffraction.py:343-347`) is the standard aplanatic rotation (`cos²φ cosθ + sin²φ`, `cosφ sinφ(cosθ−1)`, `−cosφ sinθ`); verified against the independent polar-quadrature oracle.
3. **Debye-Wolf prefactor sign and magnitude** (`vector_diffraction.py:430-433`): `(−ik/(2πf))·exp(+ikf)·dx_p²`. Low-NA absolute peak measured **6.202465e1** vs scalar Fraunhofer `π a²/(λf)` = **6.203777e1** → ratio **0.999788**.
4. **Low-NA scalar Airy limit** (NA=0.05, uniform x-pol, λ=633 nm, f=5 mm): normalised intensity vs `[2J1(v)/v]²` over r<5r₁ → **relL2 6.694e-4**, max abs deviation 4.63e-4. First-zero radius converges monotonically: **0.61638 → 0.61138 → 0.60967 λ/NA** at dx_focal = 1545/773/386 nm, vs theory 0.60976 (**rel err 2.7e-4**). No spurious warnings fired (I ran with `simplefilter('error', RuntimeWarning)`).
5. **NA=0.9 longitudinal energy fraction** (n=1, uniform aplanatic pupil, x-pol, λ=633 nm) — **measured, I did not verify a textbook value**: `|Ez|²` fraction = **0.2290** over the whole focal-plane window; **0.180–0.188** inside r<0.61λ/NA; **0.191** inside r<λ; **0.219** inside r<3λ. `|Ey|²` = 0.0133 (whole window). peak`|Ez|²`/peak`|Ex|²` = **0.135–0.150**. `Ez ≡ 0` on axis (5.9e-31 — the symmetry requirement for x-pol). Stable across (Np, dx_p, N_focal) = (512, 4 µm, 1024), (512, 4 µm, 2048), (1024, 2 µm, 2048). The central-region value sits in the 15–20 % band the prompt quotes.
6. **Axial/defocus kernel** (`vector_diffraction.py:358`) is the correct `exp(+ikz cosθ)` and is *not* affected by finding #1.
7. **RW guards** — the immersion-NA rejection (118-122) and all three diagnostics (136-147 `dx_focal` mismatch, 212-245 array-doesn't-span-pupil, 285-304 `N_focal` crop) gate correctly on actual truncation and stayed silent on every legitimate probe.
8. **RS kernel form** (`rs.py:252-253`): `h = (z/(2πr²))·exp(ikr)·(1/r − ik)` is Goodman 3-43 with the right sign; leading term `cosθ·exp(ikr)/(iλr)` — **Rayleigh-Sommerfeld I**, correct `cosθ` obliquity, correct `1/r` (not `1/r²`) power. Input pixel area **is** included (`h = h*(dx*dy)`, line 253). `r=0` cannot occur (`z > 0` hard-enforced at 182-188). At z ≥ 200 µm on the N=128/dx=1 µm probe, RS matched the exact oracle to **3.8e-8 … 4.0e-8** with power conserved to 1e-6, and **beat** ASM at long z (ASM 1.6e-4 at 1 mm, 0.28 at 3 mm from wrap-around).
9. **RS `H` cache key** (`rs.py:235-237`) is complete — `(2Ny, 2Nx, dy, dx, λ, z, bandlimit, dtype, 'RS')` covers every quantity the kernel depends on. Not stale, and correctly restricted to the NumPy backend.
10. **Van Vleck machinery in `propagate_huygens_fresnel_with_opl_callable`** (`hf.py:352-377, 404`) is correct — and importantly **it is not missing the obliquity**, contrary to what a reading of the code suggests. For the exact spherical OPL `Φ = r/λ` the cross-Hessian determinant is `(1/(λr))²·(z²/r²)`, so `sqrt|det| = cosθ/(λr)` — the RS-I amplitude, **emergent**. Measured at the grid corner: `sqrt|det| = 7.082823e9` vs `cosθ/(λr) = 7.082940e9`, ratio **1.0000**. End-to-end on a 48×48→9×9 probe at z=200 µm it matches an exact RS-I direct sum on the same discretisation to **relL2 2.50e-5**, vs 2.63e-4 for Kirchhoff `(1+cosθ)/2` and 1.02e-3 for paraxial `1/(iλz)`. The `-1j` Maslov factor (line 404) is the correct `(2π)^{−d/2}·i^{−d/2}` for d=2 under the waves-valued Φ convention. The 2.50e-5 residual is exactly the `(h/z)² = (1e-6/2e-4)² = 2.5e-5` finite-difference truncation — i.e. the absolute `finite_diff_step=1e-6 m` default degrades as `1/z²` away from the z=50 mm scale it was tuned at (documented at `hf.py:254-259`, but worth a louder note).
11. **HFPI phase accumulation** (`hfpi.py:268-272`): directions are unit vectors so `|t|` is the true geometric path length, and `exp(ik·Δs)` is applied correctly. The docstring's claim that fringe positions and interference contrast are right is consistent with what I measured.
12. **HFPI cell-centred binning** (`hfpi.py:438-439`, `vectorial_hfpi.py:360-361`): `floor(x/dx + N/2 + 0.5)` correctly maps pixel i to `[x_i − dx/2, x_i + dx/2)` on the library's `(arange(N) − N/2)·dx` grid.
13. **`_complex_output_dtype`** (`hfpi.py:514-534`) and its use at `hfpi.py:680, 1061` / `vectorial_hfpi.py:463` correctly prevent the real-accumulator imaginary-part loss.
14. **Vectorial transverse projection** (`vectorial_hfpi.py:278-286`) is algebraically right: `E·ρ̂ = Ex·L + Ey·M`, `E_t = E − (E·ρ̂)ρ̂`.
15. **`mhs.asm_subdomain` pitch guard** (382-388) correctly rejects `in.dx != out.dx`; `MhsPipeline.run`'s conditional `append_plane` import (234-244) does close correctly over the nested `_persist`; `_hfpi_segment_trace` correctly ANDs the traced alive flag with the incoming one (`hfpi.py:1120`).
16. **The dual return type of `propagate_huygens_fresnel_freespace`** (bare ndarray vs `(E, dx)` tuple, `hf.py:119-128`) is fragile but **is handled**: measured through `propagate(method='hf', output_dx=…)` and `output_grid={'N':…,'dx':…}`, the dispatcher returns a `PropagationResult` whose `.field` is an ndarray in every case, matching `gbd`.

---

## Unverified

- **Handedness of `polarization='circular'`** = `(1, 1j)/√2` (`vector_diffraction.py:313`, documented "right-circular"). IEEE and Born & Wolf disagree on the label and I found no library-wide convention statement to check it against. Not called a bug.
- **`mhs.from_prescription`'s `thickness_total = sum(prescription['thicknesses'])`** (`mhs.py:334-335`) — I did not trace a real prescription through it to confirm the list excludes object/image distances.
- **Whether `apply_vector_aperture_diffraction` double-counts** by applying both the scalar obliquity `0.5(cosθ_in+cosθ_out)` (line 268) and the transverse projection (278-286) when `vector_projection=True`. Plausible, but I have no oracle for this heuristic and the option is default-off.
- **First-call latency** of `propagate_huygens_fresnel_with_opl_callable`: measured 44 s and 180 s on two cold interpreters while identical warm calls take 0.026 s. I believe this is `array_namespace`'s backend import probe (`hf.py:308`), not the algorithm, but I did not isolate it — the finding-#12 numbers above are warm-timing only.
- **CuPy and JAX backends were not exercised at all.** Every backend-specific branch (`rs.py:198-205, 280-285`; `hfpi.py:480-484, 495-510`; `vectorial_hfpi.py:334-351, 374-389`) is untested by this audit.
- I did **not** run the repo's own test suite, so I cannot say how many tests currently pin the flipped Richards-Wolf orientation. `tests/unit/test_niche_s9_vector_diffraction_registration.py` is the obvious one to check before applying the #1 fix — though every even/even symmetric-pupil test will be bit-identical either way (measured 1.1e-16).

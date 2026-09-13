# VERIFY-WP-B8 — adversarial re-verification of the analysis / sources work package (A6.1–A6.3, Z3 §6.1–6.3, audit §15.9)

Verifier: VERIFY-WP-B8 (independent; did not write WP-B8).
Subject: commit **`ed40e169`** on `audit-fixes-2026-09`, its parent **`ed40e169^` = `8dab7de5`**,
the report `fixes/WP-B8_REPORT.md`, its changelog, and
`tests/unit/test_audit2609_b8_analysis_sources.py`.

Every claim below was **re-measured, not read**. Where WP-B8 used an oracle I built a
different one (a second, independent exact-rational Zernike oracle via the Jacobi
recurrence; a lag-averaged correlation estimator; my own brute-force centred Fourier sum,
Airy and Gaussian PSFs; my own chi-square). Date of every measurement: **2026-09-13**, this
host, Python 3.14.6 / NumPy 2.4.6 / SciPy 1.17.1 / JAX 0.10.1 (CuPy absent), every run under
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a time. No git
write command was run.

**Method note.** Byte-identity was measured **archive-to-archive**: `git archive ed40e169^
lumenairy` and `git archive ed40e169 lumenairy` extracted into
`…/scratchpad/verify_b8/{parent,wp}`, each imported from a child process whose `cwd` and
`PYTHONPATH` are that archive with `lumenairy.__file__` asserted, never through pytest and
never against the shared working tree (which carries four other engineers' uncommitted
edits). WP-B8's report used `git archive HEAD` at `284daccc` instead of `HEAD^`; I checked
that choice rather than accepting it — `git rev-parse 284daccc:<file>` and
`8dab7de5:<file>` are the **same blob** for all six files WP-B8 owns, so its baseline and
mine are the same library.

My matrix is **4056 named results** (3540 general + 516 Jones), designed to include the
cases WP-B8's 1125 did not: non-square `compute_otf` grids at every parity, Fortran-ordered
and strided and read-only inputs, signed zeros, `N_psf < N_pupil`, `complex64` /
integer / 0-d / complex `rho`, aliased `Ex is Ey`, mixed-precision Jones fields, and
spatially-varying matrices at three dtypes in two layouts.

---

## 1. Verdict

| # | WP-B8 claim | Verdict | My oracle and my numbers |
|---|---|---|---|
| **1a-i** | the in-place quadrant exchange is "**bit-identical by construction on every even length**", odd lengths and non-square grids included | **VERIFIED** | 117 shape × dtype cases against `fftshift(fft2(ifftshift(a)))` restated in-process: 21 square lengths 2…512 (powers of two, even non-powers, odd) plus 18 non-square incl. one-odd-one-even `(17,16)`, `(16,17)`, `(129,128)`, `(128,129)` and the degenerate `(1,1)`, `(1,8)`, `(2,3)`, `(512,2)`, each in `complex128` / `complex64` / `float64`. **0 non-identical.** Hostile layouts (F-order, strided view, padded view, read-only, ±0.0, NaN/inf, all-zeros): values identical and the input never written through. |
| **1a-ii** | peak **4.00 → 2.00** padded grids; 268.4 → 134.2 MB at oversample 4 on a 512 pupil | **VERIFIED** | `tracemalloc` on both archives, same script, warm: parent **4.00 / 4.00 / 4.00** grids (67.1 / 268.4 / 268.4 MB) → `ed40e169` **2.00 / 2.00 / 2.00** (33.6 / 134.2 / 134.2 MB) at (Np 512, ovs 2) / (512, 4) / (1024, 2); 105.8 → 65.5, 413.7 → 340.6, 402.9 → 304.4 ms. |
| **1a-iii** | `compute_otf`'s peak "does not move", floor 2.50 grids | **VERIFIED — the honest claim is the right one** | **2.50 grids on both archives** at N = 1024 and 2048 (41.9 and 167.8 MB). The report says so instead of quoting a before/after that is not there. |
| **1a-iv** | `fft2` **is** `fft(axis=-1)` then `fft(axis=-2)` | **VERIFIED** | Bitwise true on `(8,8)`, `(16,14)`, `(17,15)`, `(64,64)`, `(128,126)`, `(33,33)`; the **reversed** order is bitwise false on all six, so the split is not order-agnostic and the pin earns its place. |
| **1a-v** | the rejected chessboard identity is bit-identical "only at powers of two" | **VERIFIED-WITH-NOTES** | Re-measured over N = 2…1024 × 3 input families: bitwise at 2, 4, 8, 16, 64, 128, 256, 512, 1024; **1.7e-16…9.0e-16** relative at 100 / 192 / 384; **1.4…2.0** relative at 17 / 63 / 129. **Note:** even at a power of two it is not unconditionally bitwise — at N = 2 on a single-hot-pixel pupil the values agree to 0.0 but the **sign of zero** differs. That strengthens the rejection; the comment now says it. |
| **1b-i** | `method='mft'` reproduces the FFT lattice to **1.1e-15** of the peak | **VERIFIED** | My sweep (Np 64/128/256 × ovs 1/2/4 × 3 normalisations, my own aberrated pupil): **worst 3.4e-15**, against a derived `sqrt(N_psf)·eps ≈ 5e-15` for a length-`N_psf` DFT reduction. Same order as the report on a fixture it did not use. |
| **1b-ii** | oracles: brute-force centred Fourier sum, closed-form Gaussian, analytic Airy | **VERIFIED** | My own brute-force `exp(-2πi x_in x_out/λf)` double sum: **9.7e-16 / 7.7e-16 / 1.3e-15** at zoom 1 / 4 / **11.3**. My own closed-form Gaussian (aliasing floor 0.0, truncation 1.6e-18): **3.4e-15 / 2.5e-15 / 1.3e-15** at zoom 1 / 4 / 16. Airy via `scipy.special.j1`: see 1b-iv. |
| **1b-iii** | **`normalize='power'` on the MFT is the analytic Parseval constant, NOT an in-window rescale** — the claim I was asked to attack | **VERIFIED, and it is the load-bearing one** | Pupil energy `2.964330000000e-07`; FFT full-grid `'power'` returns a ratio of **1.000000000000000**; the MFT's analytic constant matches the FFT's empirical ratio to **6.7e-16** of the peak. On sub-fields the energy is deliberately *not* forced to 1: 256@1× → **1.000000**, 64@4× → **0.971170**, 8@8× → **0.364476**, 8@32× → **0.029143** of the pupil energy. And the intensity **density** at a fixed physical point is window-independent — an 8×8 and a 64×64 window match the 256×256 window to **5.0e-16** and **2.7e-16** of the peak. An in-window rescale would have forced every one of those fractions to 1. |
| **1b-iv** | the Airy residual is "**the same at 1× and 8× zoom** — it is the sampled aperture's, not the sampler's" | **VERIFIED — but only when measured correctly** | At **matched physical sampling** (same window, 8× the points): 300 px/diam → **1.30e-4 (1×) vs 1.48e-4 (8×)**; 600 px/diam → **8.6e-5 vs 1.02e-4**. The 8× grid's every-eighth sample equals the 1× grid to **5.6e-16**, and the MFT matches a brute-force DFT of the *same discrete aperture* at the 8× points to **1.8e-15** — so the residual is the aperture's. **Caveat worth recording:** if the "1×" grid is chosen too coarse to resolve the Airy core (one pixel ≈ 3.8 first-zero radii) the same comparison reads 1.1e-5 vs 1.4e-4 and looks like a 12× degradation. That is a sampling artefact of the comparison, and it is how I first mis-measured it. |
| **1b-v** | odd-`N` grids differ by half a pixel and warn | **VERIFIED** | Exactly one `UserWarning` at (Np 65, N_psf 128), (64, 129), (65, 65); MFT-vs-FFT relative **6.1e-16 / 0.387 / 0.573** — half-pixel where either length is odd, machine-zero where both are even. Decentred pupil **1.4e-15**, annular pupil **3.6e-16**. |
| **1b-vi** | "with `dx_psf=None` the MFT samples exactly the lattice the padded FFT delivers" | **VERIFIED-WITH-NOTES → V4, docstring fixed** | False for `N_psf < N_pupil`, where the FFT path pads only above the pupil size and silently returns an `N_pupil × N_pupil` array while reporting `λf/(N_psf·dx)` as its pitch, and the MFT honours `N_psf`. Measured at Np 32: FFT `(32,32)` vs MFT `(16,16)` and `(63,63)`, with the **same** reported `dx_psf`. Pre-existing FFT-path bug; the new unconditional claim is what I fixed. |
| **2-i** | curve and radius through a shared `profile=` are **bit-identical** to the unshared pair | **VERIFIED** | Bit-identical over `dy ∈ {dx, 2.5 dx}` × `centroid ∈ {None, (2 µm, −3 µm)}`, both functions, on my own fixtures. |
| **2-ii** | "the radius is the exact inverse of the curve", within one pixel-radius riser | **VERIFIED** | `threshold=0` raises (documented `(0,1]`); `1.0` → `r_max` with a round-trip error of **4.3e-15**. Errors vs the riser the data supplies at that threshold: 1e-12 → 9.9e-3 vs the first shell's own 9.9e-3; 0.1 → 3.4e-3 vs 8.8e-3; 0.5 → **0.0**; 0.84 → 1.6e-3 vs 1.6e-3; 0.999999 → **0.0**. Ties are real and constant: 4096 radii, **2053 distinct**, largest block 4. |
| **2-iii** | a caller-supplied profile is never written through; no cache, no aliasing | **VERIFIED** | `p_cum` byte-unchanged after `encircled_energy_radius(..., profile=)`; two calls return distinct objects; neither array shares memory with `E`; `dy=` / `centroid=` alongside `profile=` raises on **both** consumers; a zero-power field gives an empty profile and both consumers reproduce the unshared degenerate answer exactly. |
| **2-iv** | "the consumers check what they cheaply can (shape and **monotonicity of the endpoints**)" | **NOT AS DOCUMENTED → V2/V3, fixed** | No endpoint check existed. A profile with `p_cum` running 1 → 0, or 0 → 5, or a **descending** `r_sorted`, or all-NaN, was accepted and answered. Nor was the length checked against `E.size`, so a profile built from an 8×8 field was accepted by a 16×16 call. Both are O(1) and both are now done. |
| **3a** | the basis build's power memo / mask hoist is **bit-identical** | **VERIFIED** | In the 3540-result matrix: `zbasis`, `zmask`, `zdec`, `zrec` byte-identical at N = 24 / 41 × 6 / 15 / 28 modes. |
| **3b-i** | the recurrence switches at **n = 22**, nothing below it moves | **VERIFIED, exactly** | Bit-identity against a locally restated factorial sum: n = 20 → **0 of 11** `(n,m)` differ, n = 21 → **0 of 11**, n = 22 → **11 of 12** differ, n = 23 → **11 of 12**. Independently, in the archive-to-archive matrix the **only** 210 differing results out of 3540 are at n ∈ {22, 23} and **none** at n ≤ 21. |
| **3b-ii** | the boundary is derived: the sum first passes 1e-9 at n = 22 | **VERIFIED** | Two independent exact-rational oracles — the factorial sum in `fractions.Fraction`, and the **Jacobi form** `R = (−1)^k ρ^m P_k^{(m,0)}(1−2ρ²)` by its own three-term recurrence in `Fraction` (0 disagreements over n ≤ 24 × 5 rationals). Absolute error of the float64 sum: **8.9e-10 at n = 21**, **1.54e-9 at n = 22** — the report's 8.9e-10 / 1.5e-9 reproduced. `R_n^m(1) == 1.0` exactly at every `(n,m)` to n = 40. At ρ = 0, 1e-12, 1−1e-12, 1: ≤ 5.4e-15 relative. |
| **3b-iii** | stability limit "**≤ 3.9e-15 at n ≤ 32 and ≤ 3.0e-15 out to n = 40**" | **NOT AS STATED → V5, comment fixed** | On the report's own `ρ = k/128` grid the worst is **3.907e-15 for n ≤ 32 *and* for n ≤ 40** — so "≤ 3.0e-15 out to n = 40" cannot be right, a bound over a superset cannot be tighter. On 257 random ρ in [0,1] it is **1.51e-14**, and on ρ = 1 − 10^−j (j = 1…15) **2.70e-14**. The test's bar is 1e-13 and is unaffected; only the stated envelope was wrong. The recurrence stays ≤ **5.8e-14 out to n = 64**, which closes the report's deferred item 4. |
| **3c-i** | `_banded_IF_apply` matches an explicitly materialised design matrix to 1e-15, bar `4N²ε` | **VERIFIED** (and it is *not* bit-identical for a real band, which the report does not claim) | My own dense `(N², n_act²)` design matrix: **bit-identical** whenever `rows_band ≥ N` (one band), and **8.0e-16…1.7e-15** relative for `rows_band ∈ {N/2, 7, 1}`, inside `4N²ε` = 5.1e-13…1.4e-12. The shipped `rows_band` is `(512 MiB/16)/(N·n_act²·8)`, which is ≫ N at every realistic size, so the delivered `fit_phase` is the single-band arm — consistent with `dm/*` being byte-identical in my matrix. |
| **3c-ii** | the `'auto'` ceiling is inclusive and the audit's 16×16-on-512 case sits exactly on it | **VERIFIED** | `_DEFAULT_CACHE_CEILING_BYTES = 536870912` (512 MiB); `16² · 512² · 8 = 536870912` — **equal**, and `'auto'` caches. |
| **3c-iii** | the warning fires **once**, above **half** the ceiling, naming the caller's frame | **VERIFIED** | `_IF_CACHE_WARN_BYTES = 268435456` (256 MiB). 12×12 on 512 (288 MiB) warns; **11×11 (242 MiB) does not**; 17×17 (578 MiB) is not cached and does not warn; `cache_basis=False` never warns; `cache_basis=True` at 512 MiB warns. Exactly **1** warning across construction + 3 × `phase()` + `fit_phase()`. `stacklevel=3` resolves to the caller's construction line (asserted by file and line number). |
| **4-i** | the `zgemm` factorisation is exact by construction | **VERIFIED** | Against my own direct `Σ_j exp(i(k_j·r + ψ_j))/√M` replaying the same RNG draws: **3.3e-16 / 5.1e-16 / 7.4e-16 / 4.3e-16** at M = 1 / 2 / 8 / 37 on a non-square 6×7 grid with `dy ≠ dx`. Same seed → bit-identical output. |
| **4-ii** | the marginal is circular-Gaussian; exact finite-M moment `2 − 1/M` | **VERIFIED** | My own estimator: `E[I²]/E[I]²` within **0.48 / 0.50 / 0.04 / 1.52** standard errors of the exact `2 − 1/M` at M = 8 / 32 / 128 / 512. My own chi-square vs `Exp(1)`, 20 equiprobable bins, one pixel per 4σ cell (640 independent samples): **20.69 / 19.00 / 22.25 / 35.69** against the 0.1 % critical value 43.82 at 19 d.o.f. |
| **4-iii** | the realised correlation matches `exp(−d²/2σ²)` for **both** generators | **VERIFIED** | Lag-averaged over all pixel pairs and 120 realisations on N = 96, bar = 6/√(n_real·cells): s/L = 0.02 → **0.00110** (fft) / **0.00244** (modes) vs 0.01095; 0.10 → **0.01267** / **0.00491** vs 0.05477; 0.50 → **0.02733** / **0.00365** vs 0.27386. Both pass at every point and `'modes'` is **7.5× tighter at s/L = 0.5**, where the FFT's pad is most strained. Edge-to-edge \|corr\| on 400 realisations (s.e. ≈ 0.05, truth 1.4e-54): fft **0.069**, modes **0.009**. |
| **4-iv** | `'modes'` is *cheaper* than the padded FFT — the A11 ~100× penalty estimate is inverted | **VERIFIED** | Interleaved medians of 5, per realisation at σ_g = L/8: **2.82 / 7.61 / 44.71 / 189.29 ms** (fft) against **1.29 / 1.72 / 3.05 / 12.29 ms** (modes) and **1.51 / 6.04 / 24.13 / 96.49 MB** against **0.53 / 1.06 / 2.63 / 7.35 MB**, at N = 64 / 128 / 256 / 512 — **2.2×…15.4× in time, 2.8×…13.1× in peak**. |
| **4-v** | the M heuristic is `clip((L/σ)², 128, 4096)` | **VERIFIED** | 4 / 16 / 64 / 256 / 4096 / 16384 cells → M = 128 / 128 / 128 / 256 / 4096 / 4096. `pad_sigma` other than the default raises under `'modes'`; the default is accepted. |
| **4-vi** | — (not claimed) | **DEFECT → V6, fixed** | `'modes'` accepted degenerate coherence lengths the `'fft'` generator rejects: `0.0` → raw **`ZeroDivisionError`** out of `_gori_mode_count`, whose own `cells <= 0` guard can never run because a Python-float divide by zero raises first; `nan` → an **all-NaN field, silently**; `inf` and a **negative** value → a field for `\|σ_g\|`, silently. |
| **5a-i** | `geometry_dtype=` default is byte-identical; `float32` needs `complex64` | **VERIFIED** | Default byte-identical across the 3540-result matrix. `float32` + `complex128`, `float32` + default dtype, `float16`, `int32` all raise with the §2 prefix; `geometry_dtype=np.float64` explicit is bit-identical to the default. |
| **5a-ii** | tolerance "**at most 1.19e-07 of the peak** — one float32 ULP" | **UNDERSTATED → V7a, docstring fixed** | Over N ∈ {64, 512, 2048} × 3 normalisations × on/off-axis centres out to 0.6 of the grid half-width: **1.192e-07 on axis** (exactly one float32 ULP, as claimed) but **up to 3.132e-07 off axis** — 2.6× the stated bound, because `(X − x0)` cancels in single precision. |
| **5a-iii** | — (not claimed) | **DEFECT → V7b, fixed** | A **NumPy-scalar** `x0` / `y0` silently defeated the whole feature: NEP 50 makes a NumPy scalar strong, so `float32_array − np.float64(x0)` promotes back to float64. Measured at N = 1024: peak **2.00× the complex64 output with `np.float64(0.0)` against 1.50× with `0.0`** — and a *different field* for the same physical centre. |
| **5b-i** | `apply_jones_matrix` is **bit-identical** to the pre-fix expression | **VERIFIED** | 216 array-matrix cases (2 sizes × 9 field kinds incl. mixed precision, `Ex is Ey`, NaN/inf, 1e-160, −0.0 × 4 layouts incl. Fortran and strided × 3 matrices) and 42 callable cases (3 J dtypes × 2 layouts incl. the `moveaxis` one): **0 mismatches**, and `field.Ex` / `field.Ey` are never written through. Archive-to-archive: **516 / 516 byte-identical**. |
| **5b-ii** | the complex-multiply operand order must be preserved | **VERIFIED, and it matters** | On this build `u*v` ≠ `v*u` bitwise for complex128 arrays, and `scalar*arr` ≠ `arr*scalar` bitwise; `(u*v)+w == w+(u*v)` bitwise. Mutating `np.multiply(j11, Ey, out=scratch)` to `np.multiply(Ey, j11, …)` turns **25** WP-B8 tests red. |
| **5b-iii** | "aliased `out`" reachable? / the JAX path | **VERIFIED — not reachable** | `scratch` is always a fresh `j01 * Ey`, so `np.multiply(…, out=scratch)` can never alias an input; `Ex is Ey` is covered above. `JonesField.__init__` coerces through `np.asarray`, so a JAX array arrives as an `ndarray` and `_jones_mix_2x2` only ever sees NumPy — checked with JAX 0.10.1 installed: `type(f.Ex) is ndarray`, and the result is bitwise the pre-fix expression. `compute_psf` / `compute_otf` on a JAX array take the `xp.fft` fall-back (agreement 1.8e-15 / 1.4e-17 against NumPy, unchanged from the parent). |
| **5b-iv** | "**each in-place step is conditioned on the dtypes agreeing, and the mixed case keeps the original expressions**" — pinned | **TRUE OF THE CODE, NOT PINNED → V8, pin added** | Deleting the `Ex_new.dtype == scratch.dtype` guard leaves **all 256 WP-B8 tests green**. `test_b8_..._on_a_mixed_precision_field` cannot catch it: its matrix is `complex128` (`np.asarray(matrix, dtype=complex)` makes every array-form matrix so), and under NEP 50 both products promote to `complex128`, so the gate never engages and the test's docstring describes a fall-back its own inputs never reach — TESTING_STANDARDS shape **S2**. The case that *does* engage it needs a `complex64` spatially-varying matrix, where the mutation returns `complex64` instead of `complex128` with narrowed values. |
| **6** | "1125 / 1125 byte-identical" | **VERIFIED, on a larger and harder matrix** | My independent **4056** results, archive-to-archive: **3846 byte-identical, 210 different — every one of them at n ∈ {22, 23}**, i.e. the documented recurrence switch, with **zero** unexpected differences anywhere else. |
| **6b** | the pins are derived envelopes with a real fail-before | **VERIFIED** | 9 in-tree mutations of `ed40e169`, run through the WP-B8 file: drop the quadrant exchange on one axis → **79 red**; shift the MFT output centre half a pixel → **16 red**; break the Kintner seed coefficient → **6 red**; swap the `np.multiply` operand order → **25**; always-clamp a caller's `p_cum` → **25**; accept a contradicting `dy`/`centroid` → **26**; reassociate `N*R*angular` → **31**; perturb `1/√M` by 1e-10 → **31**; move the Parseval moment back after the transform → **39**. Two stayed green — the two Jones dtype guards (see V8 and §5.2). |

---

## 2. Defects found and fixed

All eight are inside the files this pass owns. Every one has a **measured** fail-before: the
fix was reverted in a copy of the tree and the named test observed red.

### V1 — `_centred_fft2` silently transposed the memory order of its result

`ifftshift` goes through `np.roll`, whose `empty_like` carries the input's layout, and the
FFT carries it through; `_centred_fft2`'s plain `a.copy()` is C-ordered. Measured on the two
archives: `compute_otf(F-ordered psf)` returns **F-contiguous on `ed40e169^` and C-contiguous
on `ed40e169`**; likewise `compute_psf(F-ordered pupil, oversample=1)`. Same values, different
buffer — a contract change invisible to a comparison that normalises with
`ascontiguousarray`, which is why the 1125-result matrix did not see it. One word:
`a.copy(order='K')`, `lumenairy/analysis/psf_mtf_otf.py:158`. **Fail-before:**
`test_verifyb8_centred_fft2_keeps_the_input_memory_order[shape0,shape1]` and
`test_verifyb8_compute_otf_keeps_a_fortran_psf_fortran` (3 red; the odd-length parameters stay
green, correctly — they take the explicit-shift path where no copy happens). Peaks unchanged
at 2.00 / 2.50 grids after the fix.

### V2 — a supplied `profile=` had no endpoint check, while two docstrings said it did

`_resolve_ee_profile`'s docstring claimed validation of "the two shapes, the two lengths and
the **endpoints**", and `encircled_energy_profile`'s Notes claimed the consumers check "shape
and **monotonicity of the endpoints**". Neither happened. Measured: a profile with
`p_cum = linspace(1, 0)` returned `ee = [1, 0.75, 0.5]`; `linspace(0, 5)` returned
`[0, 1, 1]`; a descending `r_sorted` returned `[0, 0, 1]`; all-NaN was accepted. Added the
O(1) check the docstrings promised, `psf_mtf_otf.py:912`. The slack is **derived**:
`len(p_cum) · ε`, against a worst measured `cumsum` drift of **8.8e-12** over
N = 16…2048 × {Gaussian, noise, Airy, near-delta} with `n·ε = 9.3e-10` at N = 2048 — the
drift is 23× under the slack at N = 16 (shortest reduction, worst ratio) and 105× under it at
N = 2048, while every shape the guard rejects is off by O(1). **Fail-before:**
`test_verifyb8_a_structurally_invalid_profile_is_refused` (4 red).

### V3 — a profile from a differently-sized field was accepted

`p_cum.size` was never compared with `E.size`, so a profile built on an 8×8 field was accepted
by a 16×16 call and answered the 8×8 question. Added, `psf_mtf_otf.py:905`; the empty
(zero-power) profile still short-circuits first. A profile from a *different field of the same
size* is not detectable in O(1) and is still accepted — the docstring now says exactly that
instead of implying otherwise. **Fail-before:**
`test_verifyb8_a_profile_from_a_differently_sized_field_is_refused`.

### V4 — the `method='mft'` compatibility statement was unconditional

"`dx_psf=None` … samples exactly the lattice the padded FFT would deliver" is false for
`N_psf < N_pupil`. Docstring now names the regime and which path is at fault,
`psf_mtf_otf.py:268`. Recorded as a pin rather than a behaviour change:
`test_verifyb8_mft_and_fft_shapes_diverge_below_the_pupil_size` asserts the FFT path returns
`(32,32)` where the MFT returns `(16,16)` for the same call, that both report the same
`dx_psf`, and that at `N_psf ≥ N_pupil` the two agree to < 1e-13. Fixing the FFT path would
move a default and is §4 material.

### V5 — the recurrence's stated stability envelope was wrong in two ways

`_ZERNIKE_RECURRENCE_MIN_N`'s comment and the report both said "≤ 3.9e-15 at every `(n,m)`
with n ≤ 32 and ≤ 3.0e-15 out to n = 40". The second half is internally impossible — the
n ≤ 40 worst case contains the n ≤ 32 worst case — and both halves are grid-specific.
Re-measured absolutely (|R| ≤ 1, so absolute is the natural unit): **3.907e-15 on `k/128` for
n ≤ 32 and n ≤ 40**, **1.51e-14** on 257 random ρ, **2.70e-14** on ρ = 1 − 10^−j. Comment now
quotes all three grids and calls it **3e-14 out to n = 40**, and records **5.8e-14 out to
n = 64** (which closes the report's deferred item 4). `zernike.py:92-101`; comment-only, and
the history fingerprint confirms it — `lumenairy.analysis.zernike.md` reports `OK` unchanged.

### V6 — `generator='modes'` answered for a degenerate coherence length

Measured on `ed40e169`: `coherence_length=0.0` → `ZeroDivisionError: division by zero` from
`_gori_mode_count` (while `'fft'` returns a field); `nan` → an all-NaN field with no warning
(`'fft'` raises); `inf` → a single-phasor field (`'fft'` raises `OverflowError`); a negative
value → the `|σ_g|` field with no warning (`'fft'` raises). The helper's own
`if not np.isfinite(cells) or cells <= 0` guard is unreachable for `σ_g = 0` because the
Python-float division raises first. Guarded before the divide (`sources/core.py:2140`) and
rejected at the `'modes'` branch with the §2 prefix (`:2341`). **`'fft'` is a default and is
untouched.** **Fail-before:** `test_verifyb8_modes_refuses_a_degenerate_coherence_length`
(4 red).

### V7 — `geometry_dtype=np.float32` was defeated by a NumPy-scalar centre, and its tolerance was understated

(a) Docstring said "at most 1.2e-07 of the peak"; measured **3.132e-07** off axis (N = 2048,
`normalize='power'`, centre at 0.6 of the half-width). Restated as 1.192e-07 on axis, up to
3.2e-07 off axis, with the reason (single-precision cancellation in `X − x0`),
`sources/core.py:445-452`. (b) Under NEP 50 a NumPy scalar is strong, so
`X.astype(float32) - np.float64(x0)` comes back float64: the exponent silently ran in double
precision, the feature bought nothing, and the *values differed from the Python-float call*.
Measured peak at N = 1024: **16.80 MB (2.00× the complex64 output) → 12.61 MB (1.50×)** after
coercing the centre with `float()` inside the float32 branch only, `sources/core.py:561`. The
default float64 path is untouched; the 3540-result matrix is byte-identical before and after.
**Fail-before:** `test_verifyb8_float32_geometry_survives_a_numpy_scalar_centre`.

### V8 — the Jones dtype gate was not pinned, and the test that claimed to pin it could not

See row 5b-iv. Added `test_verifyb8_apply_jones_matrix_falls_back_when_products_disagree`,
which (i) asserts the two products really do land in different dtypes on its fixture,
(ii) evaluates the narrowed accumulation locally and asserts it differs from the pre-fix
expression — so the pass arm cannot be satisfied by a build where the two coincide — and
(iii) asserts the shipped result is bitwise the pre-fix expression in both dtype and value.
Corrected the docstring of `test_b8_apply_jones_matrix_is_bit_identical_on_a_mixed_precision_field`
to say what it actually pins (that a mixed-precision field with a complex128 matrix promotes
*both* products to complex128, so the in-place path is taken and nothing narrows). No
assertion was removed. **Fail-before:** the guard mutation, which is green on the 256-test
suite and red on the 274-test one.

---

## 3. Claims re-measured and left alone

* **The chessboard rejection (§2.1 of the report) is the right call and the reasoning holds.**
  The only correction is that "bitwise identical at every power of two" is a statement about
  values, not bits — signed zeros can still differ.
* **`compute_otf`'s peak does not move.** 2.50 grids on both archives. The report says so
  rather than inventing a before/after; that is the honest reading.
* **`_banded_IF_apply` is not claimed bit-identical and is not bit-identical** for a real
  band. The report's "1e-15 relative, bar `4N²ε`" is what I measure.
* **The DM warning's `stacklevel=3` is correct.** My first check appeared to show it landing
  on the wrong line; it was landing on the line inside my own helper that constructed the DM,
  which is the caller of the constructor. A direct construction reports its own line.
* **The second guard in `_jones_mix_2x2`** (`np.result_type(j11, Ey) == scratch.dtype` and the
  broadcast-shape check) is structurally unreachable-false: `scratch` is by construction
  `j01 * Ey`, and `j01` / `j11` always share `J`'s dtype and shape. Mutating it away changes
  nothing, which is why it is unpinned. Defensive, harmless, and worth keeping if the
  construction of `scratch` ever moves — not a defect, and I did not add a test for a branch
  that cannot be reached.
* **`f ≤ 0` / `f = inf` diverge between the methods** — `'mft'` raises with a clear message,
  `'fft'` silently returns `dx_psf = 0`, negative or `inf`. Pre-existing on the FFT side; the
  MFT is the stricter and better-behaved path. Not changed (it is a default).

---

## 4. Requested changes outside my ownership

**One, and it is the same class as WP-B8's own §6.1.** `compute_psf(method='fft')` ignores
`N_psf` when `N_psf < pupil.shape[0]`: it pads only in the `N_psf > Np` branch, so it returns
an `Np × Np` array while computing `dx_psf = wavelength*f/(N_psf*dx_pupil)` and — for
`normalize='power'` — scaling by a `psf_power_area` built from that pitch, i.e. off by
`(Np/N_psf)²`. The docstring says `N_psf` is "Size of the output PSF grid". This is
pre-existing (identical on `ed40e169^`), it is inside a file I own, and I have deliberately
**not** changed it, because raising or padding there moves a default. It wants a coordinator
decision, not a quiet edit. The minimal edit, if the decision is to refuse:

```diff
--- a/lumenairy/analysis/psf_mtf_otf.py
+++ b/lumenairy/analysis/psf_mtf_otf.py
@@ (after the method / dx_psf validation, currently line ~344)
+    if int(N_psf) < Np:
+        raise ValueError(
+            f"compute_psf: N_psf={int(N_psf)} is smaller than the pupil "
+            f"({Np}x{Np}); the FFT sampler cannot crop, so it would return "
+            f"an {Np}x{Np} array while reporting the N_psf pitch.  Ask for "
+            f"N_psf >= N_pupil, or use method='mft', which samples "
+            f"N_psf points at whatever pitch you name.")
```

Everything else WP-B8 requested (`encircled_energy_profile` at the top level and in
`test_audit2609_a15b_reexports.py::_REEXPORTS`) has **already landed** in `ed40e169`;
`tests/unit/test_v4_16_0_walker_all_symmetry.py` and
`tests/unit/test_audit2609_a15b_reexports.py` are green here.

---

## 5. Follow-up

1. **The FFT path's `N_psf < N_pupil` behaviour** — §4. Pinned as a divergence, not fixed.
2. **Odd-`N` centring across the package.** WP-B8's §6.1 stands and I reproduce it
   (relative 0.387 / 0.573 at the odd grids I tried). It is a package-wide sweep, not a
   `compute_psf` edit.
3. **A profile from a different field of the same size** remains undetectable in O(1). If it
   is worth catching, the cheap key is not the content but the *provenance*: a
   `(id(E), E.shape, dx, dy, centroid)` stamp carried inside the returned tuple would cost
   nothing and refuse the mismatch — but it changes the public return type, so it is a
   signature decision.
4. **`_zernike_radial` above n = 40** — the report deferred this for oracle cost. I swept to
   **n = 64** with the exact rational oracle in about a minute: the recurrence holds
   ≤ 5.8e-14 there. The deferral can be closed; the constant needs no upper limit.
5. **The Gori generator's non-Gaussian extension** and the **direct-matrix MFT branch**
   (report §7.2-3) I did not attempt; nothing I measured argues against either.
6. **`_gori_mode_count`'s `'fft'`-side counterpart.** I guarded `'modes'` only, because
   `'fft'` at `coherence_length = 0.0` currently *succeeds* and that is a default. Someone
   should decide whether a zero coherence length ought to be legal at all; today the two
   generators disagree about it by design rather than by accident, which is at least
   documented now.

---

## 6. Tests run

| command | result | time |
|---|---|---|
| `pytest tests/unit/test_audit2609_b8_analysis_sources.py` | **274 passed** (256 WP-B8 + 18 mine) | 18.0 s |
| `pytest tests/unit/test_audit2609_a7_{detector_sh,image_plane_wfe,misc,opd_unwrap,strehl_reference}.py tests/unit/test_audit2609_verify_a7{,_wfe}.py tests/unit/test_audit2609_a11_polar_sources_infra.py tests/unit/test_audit2609_a15b_reexports.py tests/unit/test_audit2609_a17_history_lint.py` | **267 passed** | 47.3 s |
| `pytest tests/unit -k "psf or mtf or zernike or encircled or ao_ or coherence or schell or gaussian_beam or jones"` | **900 passed, 11 skipped, 0 failed** | 538.7 s |
| `python validation/run_all.py test_analysis test_ao test_coherence test_detector test_features test_sources test_polarization` | **ALL 7 files passed** | 39.1 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py` | **739 passed** | 33.7 s |
| `pytest tests/unit/test_v4_16_0_walker_all_symmetry.py tests/unit/test_s3_7_broadcast_grid.py` | **17 passed** | 1.0 s |
| `pytest --doctest-modules lumenairy/analysis/psf_mtf_otf.py lumenairy/analysis/zernike.py` | **5 passed** | 1.0 s |
| `ruff check lumenairy/analysis/ lumenairy/sources/ lumenairy/elements/polarization.py tests/unit/test_audit2609_b8_analysis_sources.py` | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | **all five of mine OK**; the one DRIFT is `lumenairy.raytrace.ray_fan.md`, WP-B9's file | — |

Out-of-pytest measurement runs (child process, `cwd` = `PYTHONPATH` = an extracted archive,
`lumenairy.__file__` asserted): `probe_bytes.py` (3540 results) and `e_item45.py`
(516 results) on `ed40e169^`, `ed40e169` and my fixed tree; `a_item1a.py` (identity,
chessboard, peak), `b_item1b_mft.py` (MFT oracles and normalisation),
`c_airy_order_ee.py` (Airy zoom, memory order, encircled energy),
`d_item3_zernike_ao.py` / `d2_abs.py` / `d3_env.py` (Zernike oracles and envelopes, DM),
`f_gori_corr.py` (correlation and cost). Mutation battery: 9 mutations of `ed40e169` and 6
reverts of my own fixes, each run through the WP-B8 file in an isolated tree.

The `-k` run reports **900 passed, 0 failed** — WP-B8 §4.1's one non-attributable failure
(`test_v5_20_12_rcwa_jones_2d_fff_nv.py`) is not in this selection and belongs to the
concurrent RCWA work package either way.

---

## 7. Files changed

- `lumenairy/analysis/psf_mtf_otf.py` — V1, V2, V3, V4 and the 1a-v comment.
- `lumenairy/analysis/zernike.py` — V5 (comment only; fingerprint unchanged, as the gate
  confirms).
- `lumenairy/sources/core.py` — V6, V7.
- `tests/unit/test_audit2609_b8_analysis_sources.py` — 18 tests added, one docstring
  corrected, nothing removed or weakened.
- `docs/history/lumenairy.analysis.psf_mtf_otf.md`,
  `docs/history/lumenairy.sources.core.md` — re-recorded with reasons, in this change.
- `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B8.md`,
  `…/VERIFY_WP-B8_CHANGELOG.md`.

`lumenairy/analysis/ao.py` and `lumenairy/elements/polarization.py` were read, measured and
mutated, but needed no change.

# POLAR-SOURCES-INFRA audit — polarization / Berreman / coatings / sources / algebra / top-level infrastructure

Repo: `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy` (v5.45.x, branch main).
All probes under `…/scratchpad/POLAR-SOURCES-INFRA/` (`p1_jones.py` … `p12_final.py`). No repo file was created, modified or deleted.

## Scope read

Read line by line:

* `lumenairy/elements/polarization.py` (1739), `elements/berreman.py` (1256), `elements/_berreman_jax.py` (592), `elements/coatings.py` (826)
* `lumenairy/sources/core.py` (3185), `sources/__init__.py` (51)
* `lumenairy/algebra/{__init__,base,primitives,apertures,from_prescription}.py` (1694)
* `lumenairy/__init__.py` (2094), `memory.py` (1177), `cache.py` (648), `user_library.py` (940), `_deprecation.py` (659), `_context.py` (365), `progress.py` (246), `_validation.py` (212), `_cache_registry.py` (178), `_logging.py` (51)
* Cross-referenced: `CONVENTIONS.md` §§3,5,6,7; `CHANGELOG.md` entries for E-M13 / P2-15 / S4-19 / A-5 / W6 / S1-11/13; `propagators/system.py` (coating bridge), `elements/rcwa/_core.py::_normalize_pol`.

Oracles used: analytic single-layer Airy `r=(r12+r23 e^{2iβ})/(1+r12 r23 e^{2iβ})` with complex `cosθ` on the `Im(n cosθ)≥0` branch; an independent Rouard/transfer-matrix TMM; textbook Jones matrices `R(θ)diag(1,e^{iδ})R(θ)ᵀ`; the Goldstein/Collett closed-form retarder Mueller; `M = A(J⊗J*)A⁻¹`; analytic HG/LG mode orthogonality; Starikov & Wolf (1982) GSM eigen-spectrum; Marcuse MFD; energy conservation `R+T+A≤1`; `tracemalloc`.

---

## Findings

### [P1] `coating_reflectance` silently returns the **p** result for `polarization='te'` (and for every unrecognised string) — `lumenairy/elements/coatings.py:112,216,257` (and the JAX twin at `:385,388`)

**What is wrong.** The polarization argument is never normalised or validated:

```python
112:  pols = ['s', 'p'] if polarization == 'avg' else [polarization]
216:  if pol == 's':  eta_arr = n_arr * cos_t_arr
219:  else:           eta_arr = n_arr / cos_t_arr        # <-- everything not exactly 's'
257:  if pol == 's':  eta_sub = n_substrate * cos_sub    # same else-branch
```

Anything that is not the literal lowercase `'s'` falls into the p branch. `CONVENTIONS.md` §7 states the opposite and names this file: *"`s` == `te` … both aliases are accepted everywhere (case-insensitive) | `rcwa.py::_normalize_pol`, `coatings.py`"*. `rcwa._core._normalize_pol` does implement that contract and raises on junk; `coatings.py` does neither.

**Evidence** (`p2_coatings.py` §D, 100 nm MgF2 on n=1.52, 60° AOI):

| `polarization=` | returned R | branch |
|---|---|---|
| `'s'` | 0.15427715 | s |
| `'p'` | 0.00322899 | p |
| `'te'` | 0.00322899 | **p** |
| `'tm'` | 0.00322899 | p |
| `'S'` | 0.00322899 | **p** |
| `'P'` | 0.00322899 | p |
| `'banana'` | 0.00322899 | **p** |
| `''` | 0.00322899 | **p** |

Reachable from the public wave path (`p9_infra.py` §I) — `propagate_through_system` forwards the string verbatim (`propagators/system.py:140-145`):

```
|t| by polarization kwarg: {'s': 0.94840576, 'p': 0.9969635, 'te': 0.9969635, 'tm': 0.9969635}
te == s ? False    te == p ? True
```

**Impact.** A `{'type': 'coating', 'polarization': 'te'}` element — the spelling the grating half of the library uses and CONVENTIONS says is accepted — applies the TM coefficient to a TE beam, with no warning. In the example above that is a 5 % amplitude / 10 % power error, and near Brewster it is order-unity (R_s = 0.154 vs R_p = 0.003). Typos are equally silent. `coating_reflectance_jax` has the identical defect, so a gradient-based AR/HR design optimising `polarization='te'` optimises the wrong polarization.

**Fix.** One line at the top of both functions:
```python
from .rcwa._core import _normalize_pol
pol_norm = {'te': 's', 'tm': 'p'}[_normalize_pol('coating_reflectance', polarization)] \
           if polarization != 'avg' else 'avg'
```
plus an explicit `raise ValueError` naming the allowed set `{'s','p','te','tm','avg'}` — the house rule for enum-valued knobs used everywhere else in this partition.

---

### [P1] Gaussian–Schell sources deliver the **circular (periodized)** Gaussian coherence kernel, not the documented `exp(-|Δr|²/(2σ_g²))` — `lumenairy/sources/core.py:2110-2140`

**What is wrong.** `_schell_phase_realizations` filters white noise with `H(k)=exp(-|k|²σ_g²/4)` on the FFT grid and inverse-FFTs it (`:2118,2138`). The two-point correlation of the result is the inverse **DFT** of `|H|²`, i.e. the *periodized* Gaussian `Σ_m exp(-|Δ+mL|²/(2σ_g²))` with `L = N·dx`, not the Gaussian itself. Nothing in the code, docstring or the module header derivation (`:1717-1730`, `:2088-2103`) mentions the window, and no guard fires.

**Evidence** (`p6_gsm_grid.py`, 40 000 realisations, N=64, dx=1 µm so L=64 µm):

| σ_g / L | max\|μ_emp − Gaussian\| | max\|μ_emp − wrapped Gaussian\| | μ(Δ=L/4): emp / Gaussian / wrapped |
|---|---|---|---|
| 0.313 | **0.2692** | 0.0023 | 0.7719 / 0.7261 / 0.7734 |
| 0.125 | 0.0082 | 0.0082 | 0.1298 / 0.1353 / 0.1353 |
| 0.031 | 0.0073 | 0.0073 | −0.0037 / 0.0000 / 0.0000 |

The realised kernel tracks the **wrapped** Gaussian to 0.002 and departs from the documented one by 0.27 (27 % of peak) at σ_g = L/3. The deterministic unit-mean-intensity normalisation is fine (`E[⟨|φ|²⟩]` = 0.9947 / 1.0006 / 0.9999).

Downstream (`p5_sources.py` §7, 20 000 realisations, N=24, w0=20 µm, σ_g=12 µm): max relative MCF error 0.1198 against the analytic `J = √(I₁I₂)·μ`. Coherent-mode spectrum (`p12`, N=20, w0=20 µm, σ_g=10 µm, 4000 realisations): library `[0.401, 0.142, 0.139, 0.0745, 0.0699, 0.0508]` vs the exact GSM `J` on the same grid `[0.392, 0.149, 0.149, 0.0568, 0.0560, 0.0560]` and Starikov & Wolf's infinite-aperture `[0.382, 0.146, 0.146, 0.0557, 0.0557, 0.0557]` — the n=2 shell's exact 3-fold degeneracy is broken and the shell mass is ~30 % high.

**Impact.** Affects `create_gaussian_schell_source`, `create_schell_model_source` and everything built on them (`PartialCoherenceMCF`, `propagate_ensemble`, partial-coherence imaging). The wrap adds *spurious long-range coherence* between opposite edges of the grid — the failure mode that most looks like a real physical result. The docstring's own guidance ("`sigma_g >> w0` approaches the coherent limit") steers users straight into the bad regime, since a typical grid is `L ≈ 4 w0`.

**Fix.** Zero-pad the noise grid by ≥ 4σ_g on each side before the filter/IFFT and crop back (cost: one larger FFT per realisation), or — minimal change — raise/warn when `sigma_g > (N·dx)/6` and document the window constraint. Exact for `σ_g ≲ L/8` as measured.

---

### [P2] `estimate_lens_memory(..., lens_model='real')` under-predicts `apply_real_lens`'s peak by 1.6×–2.8× — `lumenairy/memory.py:589,615-617,701-711`

**What is wrong.** The docstring maps `lens_model='real'` to "bare `apply_real_lens` (omits the traced final-assembly float64 arrays)" (`:616-617`) and the branch at `:704-707` drops the Newton term and scales the float64 core by `5/_LENS_F64_ARRAYS`. Measured, the bare `apply_real_lens` peak matches the **`'traced'`** model, not the `'real'` one. `apply_real_lens` has no `lens_model` kwarg, so nothing in the API tells a user which model to pick.

**Evidence** (`p8_memory.py`, `tracemalloc` peak of one `apply_real_lens` call on an N-BK7 singlet, complex128, caches warmed):

| N | measured peak | `'real'`, parallel_amp=False | `'real'`, parallel_amp=True (default) | `'traced'` (estimator default) |
|---|---|---|---|---|
| 1024 | 203.0 MB | 72.1 MB (**0.36×**) | 127.5 MB (**0.63×**) | 188.4 MB (0.93×) |
| 2048 | 809.6 MB | 288.6 MB (**0.36×**) | 510.0 MB (**0.63×**) | 753.7 MB (0.93×) |

The ratios are identical at both grid sizes, so this is a model error, not noise.

**Impact.** A pre-flight budget computed with the documented `'real'` model under-reserves by up to 2.8× — the exact failure `check_sim_memory` exists to prevent. The default (`'traced'`) call is fine (7 % conservative in the fail-safe direction only by luck; it is 7 % *under*).

**Fix.** Either make `'real'` reproduce the measured peak (drop the `5/_LENS_F64_ARRAYS` scale-down; the bare entry point retains the same float64 stack) or delete the branch and document that `apply_real_lens` is modelled by `'traced'`. Add a tracemalloc-vs-estimator regression at N=512 to pin it.

**Bonus (correct):** `estimate_asm_memory` over-predicts (N=1024: 165.3 vs 100.7 MB measured, 1.64×; N=2048: 484.9 vs 402.7 MB, 1.20×) — conservative, the fail-safe direction.

---

### [P2] `create_gaussian_beam` is the last source factory still building a dense meshgrid — 3×–5× the output array as peak — `lumenairy/sources/core.py:503-505,517,527,531`

**What is wrong.** Every sibling factory was converted to broadcast views and carries the comment `# S3-7: broadcast views, not a dense N x N grid` (`:689, :861, :993, :1055, :1247, :1330, :1690, :2279`). `create_gaussian_beam` still does

```python
503:  x = (xp.arange(Nx) - Nx / 2) * dx
505:  X, Y = xp.meshgrid(x, y)              # two dense N^2 float64 arrays
517:  E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
518:  E = E.astype(target_dtype)
527:  E = E / mx                            # out-of-place; _apply_field_normalization uses /=
```

**Evidence** (`p11_perf.py` §1, N=8192):

| dtype | time | peak | output | peak/output |
|---|---|---|---|---|
| complex128 | 5.03 s | 3221.4 MB | 1073.7 MB | **3.00×** |
| complex64 | 5.53 s | 2684.5 MB | 536.9 MB | **5.00×** |

**Impact.** ~2.1 GB of avoidable transient at N=8192 (the two meshgrid copies plus the out-of-place normalisation divide), and the complex64 request buys nothing because the float64 geometry and `exp` never shrink. `create_fiber_mode` and `Source.gaussian` route through this function, so they inherit it.

**Fix.** `X, Y = x[None, :], y[:, None]` (matching the siblings); use `E /= mx` / `E /= norm` in place (the shared `_apply_field_normalization` at `:194-223` already does, and documents why); optionally build the exponent in the target real dtype when `dtype=complex64`.

---

### [P2] Algebra `FreeSpace` opts into the legacy return contract and emits the library's own "no stable contract" warning on every far-field segment; the composite ABCD and the delivered grid disagree — `lumenairy/algebra/primitives.py:165-176`

**What is wrong.** `FreeSpace._apply` calls `propagate(..., method=self.method, return_result=False)`. With the class default `method='auto'` the dispatcher picks SAS in the far field and `propagate` raises a `UserWarning` reading *"a caller that unpacks this return has no stable contract … drop the argument — since v5.30 the DEFAULT is the shape-stable PropagationResult"*. The caller cannot act on that advice: the argument is the algebra layer's own, hard-coded for the anamorphic-dy reason documented at `:150-164`.

**Evidence** (`p10_algebra.py` §F, 4f system `FreeSpace(f)·ThinLens(f)·FreeSpace(2f)·ThinLens(f)·FreeSpace(f)`, f=200 mm, N=256, dx=8 µm):

* three `UserWarning`s per evaluation, all pointing at `algebra/primitives.py:165`;
* ABCD is exactly `[[-1,0],[0,-1]]` (magnification −1, so the field should return at the input pitch) but the delivered pitch is `dx_out = 15.45 µm` — 1.93× the input;
* physics itself is right: power ratio 1.0000, centroid +300 µm → −300 µm (correct inversion).

**Impact.** Noise the user cannot silence except by naming a method on every `FreeSpace`; and the module's selling point ("read system ABCDs *and* apply to fields") delivers a field whose transverse sampling contradicts the ABCD the same object reports.

**Fix.** Default `FreeSpace(method='asm')` (pitch-preserving, anamorphic-aware, matches the ABCD), or keep `'auto'` but consume `PropagationResult` and re-sample to the ABCD-implied pitch; either way document the pitch contract next to `.abcd`.

---

### [P3] `deprecated_alias` stacklevel is off by one — deprecation warnings name the library, not the caller — `lumenairy/_deprecation.py:470-476,549-556`

`_emit` uses `stacklevel=3` (`:476`), which from inside `_emit` resolves to the *caller of the helper*. For the direct helpers that is the public library function; for `deprecated_alias._shim` (`:551`) it is `_deprecation.py` itself. The `_emit` docstring's claim — *"`stacklevel=3` lets the warning point at the caller of the public function"* — is wrong by one frame in both cases.

**Evidence** (`p9_infra.py` §A/B): calling `la.load_zmx_prescription(...)` from a user script attributes the `DeprecationWarning` to `filename=_deprecation.py line=551`. `la.load_zmx_prescription` and `la.load_zemax_prescription_txt` (`__init__.py:1090,1096`) are the library's only live aliases, so both are affected.

**Impact.** `-W error::DeprecationWarning` tracebacks, IDE problem panes and `warnings.filterwarnings(..., module='myapp')` all point into lumenairy instead of the offending user line.

**Fix.** `_shim` should pass `stacklevel=4` (and `warn_deprecated_*` should default to 4 with the docstring corrected).

**Also measured (`p9_infra.py` §C):** the alias shim costs **7.66 µs/call** (0.09 µs raw → 7.75 µs aliased, 85×) — `warnings.warn`'s stack walk, paid on every call even after the first. Cache a "already warned" flag if either alias ever lands in a loop.

---

### [P3] `lumenairy.algebra.from_prescription` the *module* shadows the *function* — `lumenairy/algebra/__init__.py:57`, `algebra/from_prescription.py:207`

`__init__.py` imports the function under a private name (`as _from_prescription`) and attaches it only as `Operator.from_prescription`. The submodule import binds the *module* to `lumenairy.algebra.from_prescription`, so the documented-looking call fails:

```
>>> from lumenairy.algebra import from_prescription
>>> from_prescription(rx, 633e-9)
TypeError: 'module' object is not callable
```

`from_prescription.py` declares `__all__ = ['from_prescription']` and its own docstring says "Provides :func:`from_prescription`"; `la.from_prescription` does not exist. Only `la.Operator.from_prescription(rx, wl)` and the fully-qualified `lumenairy.algebra.from_prescription.from_prescription` work. Fix: rename the module (`_from_prescription.py`) or re-export the function under a non-colliding public name.

---

### [P3] `_check_2d_scalar_field` promises "2-D complex" but enforces only `ndim == 2` — `lumenairy/_validation.py:166-212`

The guard tests `getattr(E, 'ndim', None) != 2` and nothing else, while every error string it emits says *"expected 2-D complex {input_kind}"*.

**Evidence** (`p9_infra.py` §D, `p11_perf.py` §4):

| input | guard | reaches `angular_spectrum_propagate`? |
|---|---|---|
| 2-D complex128 | ACCEPTED | yes |
| 2-D float64 / int64 / bool | ACCEPTED | yes (→ complex128) |
| **2-D object dtype** | ACCEPTED | yes (→ complex128, via the object slow path) |
| **`np.matrix`** | ACCEPTED | yes (returns ndarray) |
| 1-D / 3-D / 0-D / nested list | rejected (ValueError) | — |

`np.matrix` is the dangerous one: its `*` is matrix multiplication, so any kernel that multiplies a field by a mask/phase elementwise silently computes a matrix product instead. Real-dtype input is probably intentional (an amplitude mask); object dtype and `np.matrix` are not.

**Fix.** Add `np.iscomplexobj(E) or np.issubdtype(E.dtype, np.floating)` and an explicit `isinstance(E, np.matrix)` rejection, or soften the message to match what is actually enforced.

Secondary nit: `:39` uses an absolute `from lumenairy.sources.core import …` where the rest of the package uses relative imports.

---

### [P3] Cache registry: a duplicate name silently drops the second clearer, and a failing clearer is invisible — `lumenairy/_cache_registry.py:92-100,146-154`

`register_cache_clearer` returns early on an existing name (`:93-99`) — intended as reload-idempotence, but it equally drops a *different* function registered under a colliding name, leaving that cache permanently unclearable. `clear_all_registered_caches` then swallows `ImportError/RuntimeError/AttributeError` per clearer (`:147-154`), so a caller who invoked `clear_asm_caches()` to free RAM before a big allocation gets no signal that a cache stayed full — in the module whose entire purpose is to retire the "fix N, miss N+1" pattern.

**Evidence** (`p9_infra.py` §G): registering two distinct lambdas under `'probe_dup'` and calling `clear_all_registered_caches()` executes only the first; a clearer that raises `ImportError` produces no output and no exception.

**Fix.** Warn (or raise) when the name is taken by a *different* callable; accumulate failures and emit one `RuntimeWarning` naming them.

---

### [P3] `_plane_wave_carrier` centres its grid on `nx//2` while the whole rest of the package uses `N/2` — `lumenairy/elements/polarization.py:1583-1585`

```python
1583:  xg = (np.arange(nx) - nx // 2) * dx
```
vs `apply_jones_matrix`'s callable grid (`:628-629`), every `sources/core.py` factory, and every `elements/elements.py` grid, all of which use `- Nx / 2`. Identical for even N; a half-pixel offset for odd N.

**Evidence** (`p6_gsm_grid.py` §2, nx=7, dx=1 µm): package convention `[-3.5 … +2.5] µm`; carrier grid `[-3 … +3] µm`, zero-phase sample at index 3.

**Impact.** A `JonesField` built by `jones_field_from_orders` on an odd grid is offset by `dx/2` relative to every element subsequently applied to it (apertures, spatially-varying Jones callables). Odd grids are rare, so low severity — but the two conventions sit 950 lines apart in the same file.

---

### [P3] `JonesField.propagate*` mutate in place and return `self`; two docstrings say otherwise — `lumenairy/elements/polarization.py:250-320`

`propagate_fresnel` (`:303`) and `propagate_fraunhofer` (`:313`) are documented "Returns new grid spacings"; both return `self` and rebind `self.dx/self.dy`. `propagate` (`:250-276`) documents neither the mutation nor the return. Only `sas_propagate` (`:351-355`) states the in-place convention. Measured (`p11_perf.py` §5): `jf.propagate(z, wl) is jf` → True, `jf.Ex` rebound. The caller's *original* arrays are not written (rebinding only), so this is a docs defect, not aliasing corruption.

---

### [P3] `import lumenairy` eagerly pulls the whole rigorous-solver stack and `scipy.linalg`

`python -X importtime -c "import lumenairy"` (warm, `p_importtime`) — the machine was heavily loaded so absolute times are not meaningful; the *tree* is:

```
lumenairy  →  lumenairy.analysis  →  analysis.coherence  →  elements.lenses
           →  elements  →  elements.berreman  →  elements.rcwa  →  rcwa._core
           →  lumenairy.backend  →  backend.scipy  →  scipy.linalg   (65 % of the cumulative tree)
```

128 `scipy.*` submodules and 741 modules total are imported to do `import lumenairy`. `numpy.testing._private.utils` is dragged in by `scipy._lib.array_api_compat` — scipy's doing, not lumenairy's.

Good news, verified: `matplotlib`, `jax`, `numba`, `cupy` are **not** imported (CONVENTIONS §10 honoured). The obvious win is deferring `backend.scipy` (hence `scipy.linalg`) behind a module `__getattr__`, since the rigorous solvers are a minority use of the package.

---

## Performance opportunities

Measured with `tracemalloc` peak / `time.perf_counter` min-of-3 (`p11_perf.py`). "full-grid array" = `N²·itemsize`.

| site | measurement | note |
|---|---|---|
| `create_gaussian_beam` N=8192 | peak/output **3.00×** (c128), **5.00×** (c64) | F4 above — dense meshgrid + out-of-place normalise |
| `create_laguerre_gauss` N=4096 p=3,l=2 | 6.80 s, peak/output **5.50×** | `arctan2`+`hypot` over 16.7 M points dominate; the recurrence allocates 2–3 N² arrays per step. `laguerre_generalized` **is** vectorised — the "genlaguerre per pixel" worry is unfounded |
| `create_hermite_gauss` N=4096 m=3,n=2 | 1.59 s, peak/output 2.50× | 4× faster than LG for the same grid — the LG cost is the polar transform, not the polynomial |
| `apply_jones_matrix` (existing field, N=2048) | peak **4.00** full-grid complex arrays | `J00*Ex + J01*Ey` builds 2 temporaries per component; `np.multiply(..., out=)` + `+=` would halve it |
| `apply_waveplate` N=2048 | peak 6.00 full-grid complex | includes the 2 input copies in the probe |
| `stokes_parameters` N=2048 | peak **7.00** full-grid real arrays | `abs(Ex)**2` recomputed for S0 and S1; `Ex*conj(Ey)` recomputed for S2 and S3. Compute `a=|Ex|²`, `b=|Ey|²`, `c=Ex·conj(Ey)` once → 3 arrays + 4 outputs |
| `degree_of_polarization` N=2048 | peak **8.25** full-grid real arrays | builds `safe`, three ratios, three squares, `live`, two `np.where` |
| Berreman modal / interface LRUs | `p4_bjax.py` | keys are `(eps.tobytes(), shape, Kx, Ky)` — complete; arrays frozen non-writeable; both guarded by their own locks. Correct, no action |
| `coating_reflectance` layer product | tournament `log2(n_layers)` batched matmul over the wavelength axis | already optimal for the shape |
| LG/HG `normalize='power'` | uses in-place `E /= norm` | correct dtype preservation (verified complex64 survives all three modes) |

Suggested order of work: (1) meshgrid + in-place normalise in `create_gaussian_beam`; (2) three-temporary `stokes_parameters` / `degree_of_polarization`; (3) `out=` in `apply_jones_matrix`; (4) lazy `backend.scipy`.

## Alternative algorithms / methods

**(a) Chipman polarization ray tracing (PRT) instead of 2-D Jones for the 3-D vector case.** `JonesField` is a strictly transverse `(Ex, Ey)` container; the class docstring (`polarization.py:124-139`) honestly lists the missing Ez / Debye–Wolf depolarization above NA ≈ 0.3 and the missing basis rotation in `propagate_tilted`. Chipman's 3×3 **P matrices** (*Polarized Light and Optical Systems*, ch. 9–11; Yun, Crabtree & Chipman, *Appl. Opt.* **50**, 2855 (2011)) carry the full 3-D field through a ray path as `P = Σ Oₒᵤₜ J Oᵢₙᵀ` with orthogonal basis-transfer matrices, and reduce exactly to the 2-D Jones matrix in the transverse limit. That is the principled upgrade for the high-NA and large-tilt cases the docstrings currently just disclaim, and it composes with the existing `raytrace/` machinery (each surface already produces the local s/p basis). Cost: one 3×3 complex matrix per ray/pixel instead of 2×2 (2.25×) — cheap relative to the existing per-surface Newton solve.

**(b) Gori's pseudo-mode / random-phase-screen representation for GSM sources.** For the finding above and for large mode counts, the coherent-mode decomposition of a GSM source is `O((N²)³)` to diagonalise and `O(K·N²)` to sample, whereas Gori & Santarsiero's *genuine cross-spectral density* form `W(r₁,r₂) = ∫ p(v) H*(r₁,v) H(r₂,v) dv` (Gori & Santarsiero, *Opt. Lett.* **32**, 3531 (2007); Martínez-Herrero, Mejías & Gori, *Opt. Lett.* **34**, 1399 (2009)) lets a GSM ensemble be generated as a superposition of *tilted, shifted* Gaussians with a Gaussian weight `p(v)` — no FFT, no window, and therefore **no periodic wrap**. That would fix the P1 above *and* remove the `n_realizations` × N² FFT cost. It also yields the exact Starikov–Wolf spectrum by construction rather than by sampling.

**(c) Berreman: Passler & Paarmann 4×4 scattering-matrix formalism.** The task suggested this; `berreman.py` has **already adopted it** — `_solve_core` (`:408-481`) composes layers with `_interface_smatrix_general` / `_propagation_smatrix_general` / `_redheffer_star` referencing forward modes to the layer top and backward modes to its bottom, which is exactly the Ko & Inkson / Passler–Paarmann (*JOSA B* **34**, 2128 (2017)) stabilisation. Verified stable: a 1 m thick `n = 1.5+0.01j` layer returns finite `R = 3.518e-2`, `T = 0.0` with no overflow (`p3_berreman.py` §D). The remaining literature item worth considering is Passler's **adaptive eigenvector sorting** for the *degenerate* case (their §3.2, where two eigenvalues coincide and `np.linalg.eig` returns an arbitrary basis in the degenerate subspace) — `_split_fwd_bwd` (`:170-192`) partitions by decay/propagation flag but does not re-orthogonalise inside a degenerate pair. I could not construct a failing case (the gyrotropic and fully-rotated-biaxial sweeps all conserved energy to 1e-16), so this is a robustness suggestion, not a defect.

## Code organization observations

* **`sources/core.py` is 3185 lines and ~60 % docstring/changelog prose.** The five removed-shim narratives (`:41-66`, `:1430-1470`, `:2018-2066`, `:2208-2219`, `:2266-2271`) document code that no longer exists. `_deprecation.py` has the same shape: `REMOVAL_SCHEDULE` is an empty tombstone and `NEXT_REMOVAL_VERSION` has been slipped five times (`:132-155`) with nothing scheduled behind it. Both belong in the CHANGELOG.
* **`berreman.py` runs two internal paths under opposite permittivity gauges** — `_solve_core` on raw public eps, `_offplane_oblique_solve` on conjugated internal eps (`:412-418`, `:709-728`). The comment warns about it explicitly, which is good, but the conj/negate bookkeeping at `:723-728` (modal-H must negate as well as conjugate) is the kind of thing that will break on the next refactor. A single-gauge internal representation with conversion only at the boundary would remove the hazard.
* **Same-named module and function** — `algebra/from_prescription.py::from_prescription` (F7).
* **Return-type inconsistency across the solver family** is documented in `berreman_jones_1d`'s own docstring (`:820-828`): `result[3]` is the *transmission* Jones here and the *reflection* Jones in `rcwa_jones_1d` / `pmm_jones_1d`. Documented, but a named tuple or dataclass would end it.
* **`lumenairy_context` scopes process-global state.** Nesting, exception restore and entry-time rollback all verified correct (`p9_infra.py` §E). Two threads using it concurrently is inherently unsafe by construction; I ran 400 interleaved `with` blocks from two threads setting opposing dtypes and observed **0** mismatches (§F) — the GIL plus a short body hides it — so I report it as a design property, not a demonstrated bug. A `threading.local` note in the docstring would be honest.
* **`call_progress` and `_atexit_restore`** both swallow exceptions with narrowed, documented tuples — good practice, correctly scoped.
* **`user_library._safe_eval_expression`** (`:174-325`) is a genuinely well-built allowlist AST interpreter: no string constants (so no `getattr` by name), no dunder access, `np` attribute access restricted to a curated pure-math set, calls only to whitelisted callables, `np.load` used with numpy ≥ 1.16's `allow_pickle=False` default. I tried to escape it and could not. The one residual is a resource-exhaustion path — `ast.Pow` on integer literals (`2**(10**9)`) runs unbounded in CPython's bignum — worth a magnitude cap on `Pow` operands.

## Unverified suspicions

* `_split_fwd_bwd` uses a decay/phase criterion, not the Poynting `Re(E×H*)_z` criterion, for the *native* NumPy Berreman cascade (`berreman.py:170-192`), whereas the JAX generalized path *does* use `Sz > 0` (`_berreman_jax.py:402-406`). For passive media the two agree (decay ⟺ `Im(kz) > 0`), and I could not build a physical case where they differ (lossy rotated-biaxial and Hermitian gyrotropic sweeps all conserved energy exactly). A genuinely negative-group-velocity anisotropic medium would separate them; I did not construct one.
* `lumenairy_context` thread-safety (above) — real by inspection, not demonstrated.
* `ByteBudgetedLRU.get_or_compute` has a benign TOCTOU (two threads can both miss and both compute); documented as "compute runs unlocked", costs duplicate work only.

## Checked and found correct

**Polarization (`polarization.py`)**
* Jones matrices vs textbook for 8 angles (`p1_jones.py` §1): polarizer `R diag(1,0) Rᵀ` **0.0**; QWP/HWP/δ=0.7 retarder `R diag(1,e^{+iδ}) Rᵀ` **1.11e-16**; rotator `[[c,−s],[s,c]]` **0.0**.
* Retardance sign: slow axis `exp(+i·δ)` per CONVENTIONS §7; QWP fast axis at **+45°** on x-pol → **S3 = −1.0000000000000004** ('left'), at −45° → **+1.0000000000000004**. Matches the table row exactly.
* `create_circular_polarized('right')` = `(1,+i)/√2` with S3 = +1; `'left'` = `(1,−i)/√2` with S3 = −1. Closed alias set; junk raises.
* Stokes formulas: `S3 = −2 Im(Ex conj Ey)` as documented; `polarization_ellipse` round-trips `(χ, ψ)` exactly (15 cases, |error| < 1e-12 including the χ=±0.7 domain edge).
* **Mueller cross-check settled by measurement** (`p11_perf.py` §6): the Mueller implied by this module's (Jones, Stokes) pair equals the Goldstein/Collett closed-form retarder Mueller with the `(S1,S2)↔S3` block negated — `|M_lib − D G D| = 0.0 / 4.4e-16 / 2.2e-16` at (θ,δ) = (0, π/2), (π/4, π/2), (0.3, 1.1), versus `|M_lib − G| = 2.00 / 2.00 / 1.47`. The module docstring's "one visible consequence" claim (`:50-57`) and the CONVENTIONS §7 row are **accurate**.
* Jones↔Mueller rotation sign: `apply_rotator(θ)` gives the Stokes rotation by `2θ` with `cos2θ = 0.738469`, `sin2θ = 0.674288` in the `(S1,S2)` block — consistent active rotation in both calculi.
* PBS power conservation exact (`|E_t|²+|E_r|² = 1.0`); ER < 1 correctly raises.
* `JonesField` propagation dispatches the same ASM to both components with a shared `H` build above N=512; the missing Ez at high NA and the missing basis rotation at large tilt are documented at `:124-139`, `:288-293`, `:379-383`.

**Coatings (`coatings.py`)**
* Characteristic-matrix TMM vs the analytic Airy oracle, quarter-wave MgF2 on n=1.52, 0°/45°/70°, s and p: `|ΔR| ≤ 1.1e-16`, `|ΔT| ≤ 8.9e-16` (`p2_coatings.py` §A).
* 50 nm Au (`n = 0.27+2.78j`) on glass, same angles/pols: `|ΔR| ≤ 2.2e-16`, `|ΔT| ≤ 1.1e-16`; `R+T < 1` correctly (absorption).
* **p-transmittance with an absorbing exit medium** (`p12_final.py` §2): `n_sub ∈ {4+0.05j, 1.5+0.5j, 0.27+2.78j}` × {0°, 55°} × {s, p} — `|ΔR| ≤ 6.7e-16`, `|ΔT| ≤ 7.8e-16` against `Re(n₃* cosθ₃)/Re(n₁* cosθ₁)·|t_p|²`. The Macleod `Re(η_sub)/Re(η_amb)` form is exactly equivalent. This was the specific concern raised in the brief; it is correct.
* TIR: glass→air, `n_amb=1.52`, at 30°/41.14°(critical)/45°/60°, both pols — `R = 1.0000000000`, `T = 0`, `R+T = 1` exactly past critical; no cap, no warning. The complex-`cosθ` branch `Im(n cosθ) ≥ 0` is right.
* **r_p sign convention is self-consistent across the family.** `coatings` uses the Macleod tilted admittance `η_p = n/cosθ`, so its `r_p = −r_p^Fresnel` and `r_s = r_p` at normal incidence (measured: both `−0.206349` for n=1.52). Berreman's lab-frame `jones_r[0,0]` carries the **same** sign — `arg(Jr[0,0] / (−r_p^Fresnel)) ≤ 1.8e-15` at 0°/20°/45°/70° (`p3_berreman.py` §A). The two families agree. *Doc nit worth a line:* nothing says `phase_r` for p is π from the textbook Fresnel convention, and `propagate_through_system`'s `'reflection'` port applies `√R·exp(i·phase_r)` directly.
* `broadband_ar_v_coat` quarter-quarter admittance match, `quarter_wave_ar`'s air-ambient caveat, the Sellmeier registry ranges and the NaN/negative-wavelength guards all read correct.

**Berreman (`berreman.py`, `_berreman_jax.py`)**
* Isotropic reduction vs an independent Rouard TMM, two-layer stack on n=1.52, 0°/20°/45°/70° (`p3_berreman.py` §A): `|r_s|`, `|r_p|` agree to all printed digits; phase agrees to `≤ 3.1e-15` rad; `R+T = [1., 1.]` exactly; `R[0]`↔p, `R[1]`↔s ordering correct.
* **Cross-family waveplate check** (§B): a uniaxial slab `eps = R_z(45°) diag(n_o², n_e², n_o²) R_z(−45°)`, `d = λ/(4Δn)`, gives `jones_t` mapping x-pol to `Ey/Ex = −0.0799 − 0.9968i`, `S3 = −0.9968`; `apply_waveplate(QWP, +45°)` gives `S3 = −1.0000`. Same handedness — the element and solver families agree, as CONVENTIONS §7 requires. (The 0.3 % residual is the index-mismatched slab's Fabry–Pérot, not a convention error.)
* Lossy fully-rotated biaxial (`rot_z(0.7)·rot_x(0.4)` on `diag((1.7+iκ)², (1.5+0.5iκ)², (1.9+2iκ)²)`), κ ∈ {0, 0.02, 0.2, 1.0} × θ ∈ {0°, 35°, 65°} × φ=0.3: `R+T ≤ 1` in all 12 cases, exactly 1.0 when lossless. Hermitian gyrotropic (`eps_xy = ±0.05i`): `R+T = 1` exactly at 0° and 40°.
* Thick/lossy layer: `d ∈ {1 µm, 100 µm, 1 cm, 1 m}` of `n = 1.5+0.01j` — all finite, `T → 0` smoothly. No transfer-matrix overflow.
* **NumPy↔JAX parity** (`p4_bjax.py`): in-plane uniaxial, OOP tilted director, OOP lossy × θ ∈ {0°, 35°} × φ ∈ {0°, 30°} — `dR ≤ 3.5e-16`, `dT ≤ 4.1e-15`, `dJr ≤ 6.7e-16`, `dJt ≤ 3.7e-15`.
* **Native vs generalized (Li-2003) path** on the same OOP-oblique slab: `R`,`T` identical to all digits, `max|ΔJt| = 1.0e-15`, `max|ΔJr| = 5.1e-16`. The W6 comment's refutation of the old "~2 %" claim is confirmed.
* `jax.grad` finiteness: `dR/d(thickness) = −1447.52` (finite); `d(R+T)/d(angle) = −2.0e-16` for a lossless stack (correctly ~0). The non-differentiable mode sort is handled by a stable `jnp.argsort` on a boolean forwardness key with the permutation treated as constant — gradients flow through the gathered eigenpairs.
* Both LRU caches: complete keys (`eps.tobytes()` + shape + Kx + Ky, tagged namespace for the OOP variant), frozen non-writeable on hand-out (`_freeze`), lock covers get and put, registered with the central clearer.

**Sources (`sources/core.py`)**
* Gaussian waist convention exact: `I(w0)/I(0) = 0.135335 = e^{−2}`, `|E(w0)| = 0.367879 = e^{−1}`.
* HG orthonormality at N=512: Gram diagonal `[1,1,1,1,1,1]`, max off-diagonal **6.1e-17**. LG (p ∈ {0,1} × l ∈ {−2..2}): diagonal all 1, max off-diagonal **1.2e-16**. `laguerre_generalized` argument is `2r²/w0²` and the `rho**|l|` prefactor is correct.
* Tilted plane wave sign: `angle_x = +3°`, ASM to z = 3 mm → centroid **+157.224 µm** vs `z·tan(3°) = +157.223 µm`; untilted control **−0.000 µm**. Propagates toward +x for θx>0, as documented. Nyquist and evanescent guards both fire correctly.
* Point source sign: `z0 > 0` (documented converging) propagated `+z0` concentrates by **9700×** at the exact grid centre; `z0 < 0` stays diverging (ratio 0.99). `exp(+ikr)/r` diverging / `exp(−ikr)/r` converging under `exp(+ikz)` — correct. The `r_floor` = half pixel-diagonal clamp and the `|z0| < dx` warning are sound.
* `rng` contract (CONVENTIONS §3): `rng=1234` reproducible and **bit-identical** to `rng=np.random.default_rng(1234)`; `RandomState` unwrapping and the non-NumPy-backend rejection read correct.
* GSM/Schell normalisation: deterministic `E[⟨|φ|²⟩] = 1` (measured 0.9947–1.0006), not the biased per-realisation form — the v5.4.6 P3-10 fix is correct and load-bearing.
* **`dy` threading (CONVENTIONS §5): all 12 factories pass** (`p12_final.py` §1) — `create_gaussian_beam`, `hermite_gauss`, `laguerre_gauss`, `tilted_plane_wave`, `point_source`, `top_hat_beam`, `annular_beam`, `bessel_beam`, `fiber_mode`, `led_source`, `gaussian_schell_source`, `annular_incoherent_source` all honour a distinct `dy` in both the y-axis and the field.
* dtype: `complex64` survives all three `normalize` modes in `create_gaussian_beam`, `create_hermite_gauss` and `create_top_hat_beam` (the in-place `/=` discipline and the float32-norm chain both hold).
* Bessel `k_r = k sin(cone)` places the first `J₀` zero at the analytic radius; top-hat `normalize='power'` integrates to exactly 1.0; `create_multi_field_sources` returns the documented `(E, ax, ay)` list plus shared axes and rejects an empty angle list.
* `create_fiber_mode` is a pure MFD-Gaussian with `w0 = MFD/2` (verified against the Marcuse `0.65 + 1.619V^{-1.5} + 2.879V^{-6}` MFD to within one pixel). It is **not** an LP01 `J₀/K₀` boundary-match solve — but the docstring says so explicitly (`:1351-1377`: "`na` … does NOT set the divergence and does NOT affect the returned field"), so this is a stated scope limit, not a defect.

**Algebra (`algebra/`)**
* `Operator.from_prescription(...).abcd` vs `raytrace.seidel.system_abcd` on the same prescription: **max |ΔABCD| = 0.000e+00** for a singlet, a meniscus (N-SF11) and a cemented doublet; EFLs identical to 9 decimals.
* Thick-lens matrix: the composed chain equals the analytic Welford `L₂·T(d/n)·L₁` to **0.0**, and its EFL equals the lensmaker `1/[(n−1)(1/R₁−1/R₂+(n−1)d/(nR₁R₂))] = 0.049375233 m` exactly.
* Composition order: `(ThinLens·FreeSpace).abcd == L @ D` exactly — `A*B` means "apply B first", matching the documented `system_abcd` convention.
* 4f identity exact: `[[-1,0],[0,-1]]`, `max|M+I| = 0.0`. Field application conserves power to 1.0000 and inverts the centroid (+300 µm → −300 µm).
* Anamorphic: `CylindricalLens(f_x=0.1)` carries `abcd_x = [[1,0],[−10,1]]`, `abcd_y = I`; composites keep them separate and `.abcd` correctly raises. `Magnify(a).abcd = [[1/a,0],[0,a]]` (the Nazarathy–Shamir `V[a]` scaling convention); `FourierTransform(f).abcd = [[0,f],[−1/f,0]]`.
* `GaussianAperture` docstring's 1/e amplitude and 1/e² intensity radii (both `σ√2`) are arithmetically correct; the `Aperture` annular default-obstruction guard (v5.4.6 F-16) is right.

**Infrastructure**
* `cache.py` byte accounting is correct in both directions (`p9_infra.py` §H): `deep_nbytes` on a 16-byte view of an 8 MB base returns **8 388 608** (the retained buffer, not the window); `deep_nbytes((a, a))` returns `8 388 672` not `2×`; a value larger than the ceiling is skipped (`put` → False) rather than thrashed. Lock covers get, put, set_budget and the cross-cache global evictor; the registry is a `WeakSet` so dead caches stop counting.
* `user_library.py` is **not** a code-execution hazard: no `pickle.load`, no `exec`, no `importlib` of user paths; `np.load` relies on numpy's `allow_pickle=False` default; the phase-mask expression goes through the allowlist AST interpreter, not `eval`.
* `_safe_name` path handling: no traversal escape. Tested `'..'`, `'../../evil'`, `'..\\..\\evil'`, `'C:pwned'`, `'C:/abs/evil'`, `'\\\\server\\share\\x'`, `'con'`, `'.hidden'`, `''` — every result resolves inside the library directory (`p7_paths.py`). `'C:pwned'` is safe because pathlib appends a same-drive drive-relative component. The many-to-one collision risk is already warned about.
* `_context.py`: entry-time rollback on a mid-apply setter failure, `finally`-restore on exception, and nesting all verified (`p9_infra.py` §E — `complex64` outer / `complex128` inner / restore / raise → base dtype restored).
* `progress.py` `call_progress` narrowed-except is documented and matches the code; `ProgressScaler`'s dual 2-arg/3-arg calling convention is coherent.
* `_logging.py` installs a `NullHandler` on the `lumenairy` root — library import is silent, correct idiom.
* `lumenairy/__init__.py` `__getattr__`/`__dir__` live-forwarding of `DEFAULT_COMPLEX_DTYPE` / `DEFAULT_REAL_DTYPE` / `DEFAULT_WAVE_PROPAGATOR` / `DEFAULT_DY` (deleting the import-time snapshots so lookup falls through) is the right pattern and works.

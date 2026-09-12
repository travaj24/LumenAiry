# TR-SIBLINGS audit — traced-lens siblings (`_lens_traced_uniform`, `_lens_traced_multibranch`, `_lens_imap`, `_traced_flags`, `_lens_jax`)

Repo `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, v5.45.0, `main` @ `a1ff1e6e`.
All scratch under `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/`.
No repository file was created, modified or deleted.

## Scope read (files + line ranges actually read; what I did NOT get to)

Read in full, line by line:

* `lumenairy/elements/_lens_traced_uniform.py` 1–946
* `lumenairy/elements/_lens_traced_multibranch.py` 1–756
* `lumenairy/elements/_lens_imap.py` 1–1841
* `lumenairy/elements/_traced_flags.py` 1–476
* `lumenairy/elements/_lens_jax.py` 1–954

Read as needed to confirm findings (dependencies, allowed by the brief):

* `lumenairy/elements/_lens_traced.py` 6649–6700, 7646–7680, 7953–7975, 8200–8270, 9570–9600, 11016–11050 (the exit-vertex correction, the `output_plane_distance` contract, the `caustic=` dispatch, the imap gate)
* `lumenairy/elements/lenses_maslov.py::_fold_airy_eval` (959–986)
* `lumenairy/raytrace/*` — `surfaces_from_prescription`, `trace_jax_with_params` signature/docstring only
* `tests/conftest.py` 217–305 (the C11 leak guard), `tests/unit/test_niche_k1_kmah_caustic.py` 145–175, `tests/unit/test_v5_21_lens_accuracy_extensions.py` 590–630

**Not reached:** the Pearcey cusp path (`_trace_meridional_cusp` / `_cusp_geometry_from_branches` / `_build_pearcey_cusp_field`) was read but only *statically* verified — I could not construct a converging `n_turn == 2` fixture in the time available, so its numerical correctness is unverified (see "Unverified suspicions"). `_lens_jax`'s `amplitude='analytic'` callback leg (`_amp_callback_jax_linear`) was read but not exercised under `jax.grad` (reverse mode through `pure_callback` inside `custom_jvp` is the specific thing I did not test). `_lens_imap`'s numba kernel was measured but not compared bit-for-bit against its NumPy fallback.

---

## Findings

### **[P0] `_lens_traced_multibranch` (and therefore `caustic='multibranch'`/`'uniform'`) never propagates the exit rays to the exit-vertex plane — the "output plane" is really the last surface's sag surface** — `lumenairy/elements/_lens_traced_multibranch.py:172-182`

**What is wrong.** `_trace_launch_grid` takes `rt.trace(...).image_rays` and advances each ray by `t = output_plane_distance / N_z`:

```python
Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
t = output_plane_distance / Nz
x_out = ex.x + t * ex.L
y_out = ex.y + t * ex.M
opl = (ex.opd + float(output_plane_n) * t + (L0 * Xi + M0 * Yi).ravel())
```

`ex` is NOT at the exit vertex: `rt.trace` leaves each ray at its intersection with the last surface, i.e. at `z = sag(ρ)`. Measured directly: for `R2 = -25 mm` a ray at ρ = 6.44 mm exits at `ex.z = -8.424e-4 m`. So the construction evaluates every ray at `z = sag(ρ) + output_plane_distance` — a *ray-dependent* longitudinal position, not a plane — while the public docstring (`_lens_traced_multibranch.py:346-348`, and `_lens_traced.py:7663-7667`) promises "the output plane `output_plane_distance` past the prescription's exit vertex".

Both siblings that do this correctly are in the same family and carry the fix with a comment explaining its sign: `_lens_traced.py:9581-9589` ("Fix: propagate each ray from its current sag position to z = 0 in the exit medium… Missing this is what made the first draft of this function disagree with Newton by ~343 nm RMS") and `_lens_jax.py:558-562`. A `grep` for `t_to_vertex` scores **8 hits in `_lens_traced.py`, 8 in `_lens_jax.py`, 0 in `_lens_traced_multibranch.py`, 0 in `_lens_traced_uniform.py`**.

`_lens_traced_uniform.py:225-229` (`_trace_meridional_fold`) and `:589-593` (`_trace_meridional_cusp`) repeat the identical construction, so the fold radius `r_c`, `kappa` and the mean-eikonal fit are all resolved on the same wrong surface.

Because the branch phases and positions are *both* taken at that wrong point, the field is the correct geometric field on a curved surface, mislabelled as a plane — and worse, two branches arriving at nominally the same `(x, y)` come from different ρ, hence different `sag(ρ)`, hence **different z**: the multi-branch interference is computed between fields at different longitudinal positions.

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/repro_vertex.py` — 401-node meridional row out of `_trace_launch_grid` vs an independent vector-Snell trace I wrote (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/oracle.py`, agrees with `rt.trace` to 8.7e-19 m / 1.4e-17 m OPL on a flat-rear lens):

```
== plano-convex, FLAT rear (sag == 0): no error expected ==
R2=inf  d_out=0                  |dx_out| max 4.337e-13 um   OPD err max 2.952e-12 waves
R2=inf  d_out=45mm               |dx_out| max 2.168e-12 um   OPD err max 4.724e-11 waves
== curved rear surface (sag != 0) ==
R1=inf R2=-25mm  ap20 d_out=0    |dx_out| max     476.8 um   OPD err max      3501 waves
R1=inf R2=-25mm  ap20 d_out=40mm |dx_out| max     476.8 um   OPD err max      3501 waves
R1=inf R2=-100mm ap20 d_out=0    |dx_out| max     24.59 um   OPD err max     820.3 waves
biconvex R=+-50mm ap12 d_out=0   |dx_out| max     40.57 um   OPD err max       568 waves
```

End-to-end on the public API (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/repro_vertex_field.py`), comparing `apply_real_lens_traced(caustic='multibranch')` against the single-valued `apply_real_lens_traced` at the *same* (exit-vertex) plane, N=512, dx=25 µm, Gaussian w=3.5 mm:

```
FLAT rear (R2=inf)                phase diff rms 0.0050 rad (0.001 waves)  max 0.0207 rad
CURVED rear (R1=inf,R2=-100mm)    phase diff rms 1.7984 rad (0.286 waves)  max 3.1415 rad
CURVED rear (biconvex +-60mm)     phase diff rms 1.8100 rad (0.288 waves)  max 3.1416 rad
```

1.80 rad rms with max = π is a fully decorrelated (wrapped) phase.

**Why ~250 audits missed it.** Every physics fixture for this layer uses a plano-convex singlet with a **flat last surface**, where the sag is identically zero: `test_niche_k1_kmah_caustic.py::_caustic_prescription` (`radius 2.7e-3`, then `float('inf')`), the same file's `caustic_fold_ref` npz, `test_niche_k4_uniform_caustic.py` (same), and `test_v5_21_lens_accuracy_extensions.py::_mini_fast_singlet` (`2.0e-3`, `inf`, `inf`). I ran `pytest tests/unit/test_v5_21_lens_accuracy_extensions.py -k "multibranch or uniform or ludwig"` → **8 passed** on the unfixed code.

**Impact.** Silently wrong physics on a default path: any prescription whose last surface is curved (every biconvex singlet, every cemented doublet, every catalogue lens that is not plano-rear) gets a multibranch/uniform field with the wrong transverse map (hundreds of µm), the wrong OPL (hundreds to thousands of waves) and mutually inconsistent branch phases. `output_plane_distance=0.0` — the *default* — is affected exactly as much as a through-focus plane, because the error is the sag, not the propagation.

**Fix.** Apply the `_lens_traced.py:9581-9589` block to `ex` inside `_trace_launch_grid` *before* the dict is assembled (so `x_exit`/`y_exit`, which `_kmah_free_leg` differentiates for `Q0`, are corrected too):

```python
from ..glass import get_glass_index
n_exit = get_glass_index(surfaces[-1].glass_after, wavelength)
with np.errstate(divide='ignore', invalid='ignore'):
    t0 = np.where(ex.alive & (np.abs(ex.N) > 1e-30), -ex.z / ex.N, 0.0)
ex.opd = ex.opd + n_exit * t0
ex.x = ex.x + ex.L * t0
ex.y = ex.y + ex.M * t0
ex.z = np.zeros_like(ex.z)
```

and the same three lines in `_lens_traced_uniform._trace_meridional_fold` / `_trace_meridional_cusp`. Better still, factor the block into one shared helper — it is already written out four times in `_lens_traced.py` and twice in `_lens_jax.py`. Add a regression fixture with a **biconvex** prescription (the flat-rear fixtures cannot see this class of defect at all).

---

### **[P1] `_lens_jax` never enforces `jax_enable_x64`, so under JAX's default configuration the whole traced path silently runs in float32 — measured ~10⁶× loss of OPD accuracy** — `lumenairy/elements/_lens_jax.py:483, 638-652, 915-923`

**What is wrong.** `E_in = jnp.asarray(E_in)` (line 483) truncates a `complex128` input to `complex64` whenever `jax_enable_x64` is off (JAX's default, which lumenairy never sets). Everything downstream — `xs_in`, the traced `opd`, the Chebyshev design and `lstsq`, the Newton inversion, `x_wave`, `k0 * opl_map` — is then float32. The v4.13.0 "audit L2" comment at :638-646 says the module deliberately does *not* read `jax.config.jax_enable_x64` and instead "unif[ies] on the library-wide default dtype": but `_resolve_jax_complex_dtype(E_in.dtype)` only chooses the **output** dtype, and it is called at line 651 *after* line 483 has already truncated `E_in`, so the v4.14.0 "pass `E_in.dtype` so the input dtype is honoured" fix faithfully honours the already-truncated dtype. Neither fix touches the compute precision.

The library has a house rule for exactly this and `_lens_jax` is the exception: `lumenairy/elements/rcwa/_core.py:1088-1109` *raises* ("JAX would silently truncate … yielding quietly WRONG efficiencies and gradients") and `lumenairy/optimize/jax_merits.py:26-68` auto-enables x64 with a warning.

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/jax_x64_quant.py`, N=256, dx=20 µm, λ=587.6 nm, N-BK7 plano-convex, OPD scored per pixel against my independent exact ray trace:

| prescription | total OPL | default (x64 OFF) | x64 ON | ratio |
|---|---|---|---|---|
| singlet only | 4.55 mm | **1.51e-04 waves rms** (5.7e-4 max) | 2.06e-10 | 7.3e5 |
| singlet + 200 mm gap | 0.205 m | **1.81e-03 waves rms** (3.5e-3 max) | 3.10e-09 | 5.8e5 |
| singlet + 1 m gap | 1.006 m | **7.53e-04 waves rms** (1.9e-3 max) | 1.64e-09 | 4.6e5 |

Output dtype is `complex64` in every default-config run even though the input was `complex128`. λ/550 on a 200 mm system is 16× *coarser* than the λ/9000 parity bar `_lens_imap` quotes for the same library's traced chain.

**Impact.** Anyone calling `apply_real_lens_traced_jax` / `apply_real_lens_maslov_jax` without having separately set `jax.config.update('jax_enable_x64', True)` gets a phase screen ~10⁶× less accurate than the function's own design point, with no warning and with a silently downgraded output dtype. Gradients inherit the same truncation.

**Fix.** At the top of both entry points, do what `rcwa/_core.py` does: read `jax.config.read('jax_enable_x64')` and raise (or warn-and-enable, per `jax_merits._ensure_jax_x64`) when it is off and the caller's array is 64-bit. At minimum, refuse to silently downcast a `complex128` input.

---

### **[P1] `jax.jit` is impossible on the default (static-geometry) path of both JAX entry points** — `lumenairy/elements/_lens_jax.py:616-621` and `:867-872`

**What is wrong.** The initial-guess magnification is computed with Python `float()` on traced arrays:

```python
dx_out_x = float(x_out_grid[i_axis + di, i_axis] - x_out_grid[i_axis - di, i_axis]) / (2.0 * di * dx_in)
```

Under `jax.jit`, `x_out_grid` is a tracer and `float()` raises. The `_diff_geom` branch (lines 601-614) already has the tracer-safe form with `lax.stop_gradient`; the `else` branch does not. `apply_real_lens_maslov_jax` has only the unsafe form (no `_diff_geom` support at all).

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/jax_grad.py`:

```
jax.jit FAILED: ConcretizationTypeError: Abstract tracer value encountered where concrete
value is expected: traced array with shape float64[] … The problem arose with the `float` function.
```

and `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/jax_radii.py` confirms the *same call with `radii=` supplied* jits fine (compile+run 2.94 s, steady 1.08 s/call). Eager at N=128 is 1.038 s/call; at N=192 with `radii` it is 1.08 s/call — i.e. jit buys ~nothing here either, because the cost is dominated by the traced fan and the `(order+1, N, N)` Chebyshev stacks rather than dispatch.

**Impact.** The docstring claims "Multi-process Newton parallelism … are not replicated; JAX's vmap+JIT replaces the first" (`:419-423`). That is false for the default configuration: neither `jit` nor anything built on it (`vmap` of a jitted wrapper, `jax.lax.scan` over field points, `jaxopt`) works unless the caller opts into differentiable geometry.

**Fix.** Use the `_diff_geom` branch's tracer-safe expression unconditionally (it is already written, 8 lines above) and delete the `float()` variant; `lax.stop_gradient` makes it a no-op for gradients on the static path. Mirror it into `apply_real_lens_maslov_jax`.

---

### **[P2] `build_inverse_map`'s SHA-256 key omits every flag that changes the least-squares arithmetic, so `DETERMINISTIC_TRACED_FIT=False` (a documented bit-for-bit fail-before) is defeated by a cache hit** — `lumenairy/elements/_lens_imap.py:1105-1111`, `:1546-1567`

**What is wrong.** `_imap_key` hashes 12 scalars plus every input array, and its own docstring says the key must carry "the library version and **the flag values that select the branch**". But the solve is `_solve_lstsq_thread_safe(A, B, deterministic=_det_traced())` where `_det_traced()` reads `_lens_traced.DETERMINISTIC_TRACED_FIT` **at call time** (:1554) — and that flag, plus `LSTSQ_CONDITIONING_STEPDOWN`, `_DET_REFINE_STEPS`, `_DET_EINSUM_MIN_TERMS`, `_DET_EINSUM_BLOCK_ROWS`, `_DET_GRAM_TILE_BYTES`, `_LSTSQ_GRAM_RCOND_MIN`, `_LSTSQ_RESID_MARGIN`, are none of them in the key. `_traced_flags.py:253-261` states the contract this breaks verbatim: *"False restores the G = A.T @ A / rhs = A.T @ b route for the traced chain EXACTLY, bit for bit, and is the fail-before for the whole layer."*

`tests/conftest.py` restores leaked flags after each test (`_LEAK_GUARD_MODULES` includes `_lens_imap`) but **never drains `_IMAP_CACHE`** — `grep` for `inverse_map_cache_clear|clear_all_registered_caches` in `tests/conftest.py` returns nothing. So a map built inside `test_niche_d15_deterministic_traced_fit.py` (5 direct flag assignments) survives into every later test in the shard.

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/imap_cache.py`, real 129×129 singlet congruence, degree 14 / 120 terms / 16 641 samples:

```
== A) is DETERMINISTIC_TRACED_FIT in the cache key? ==
   cached flags: False True True
   key(det=True) == key(det=False)? True
   SAME OBJECT served for det=False? True
   coef(det=True) vs cold coef(det=False): max abs 5.724e-14  rel 6.164e-14  bit-identical=False
== B) LSTSQ_CONDITIONING_STEPDOWN in the key? ==
   same key? True  same object? True
```

Honest magnitude: on this well-conditioned fixture the two routes differ by 6.2e-14 relative (and `_DET_REFINE_STEPS = 0` changes nothing further — `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/final_batch.py` §D). So this is a **contract** defect, not a large numerical one *on this fixture*; D15's own note records 1681–2138× least-squares-residual differences on the production 120-term fits, which is the regime where it would matter.

**Impact.** `traced_flags(DETERMINISTIC_TRACED_FIT=False)` — the documented fail-before device, and the exact "intervention that is not an intervention" failure `_traced_flags.py`'s header was written to close — silently returns the D15 map on the second and every later call in a process. Same for the C13 step-down.

**Fix.** Add the seven constants to the `repr((...))` tuple at `:1105-1111` (they are cheap scalars), and register `inverse_map_cache_clear` in the `tests/conftest.py` per-test teardown next to the leak guard.

---

### **[P2] `apply_real_lens_traced_uniform` evaluates the CFU Airy kernel on EVERY pixel outside `r_c`, when the physical tail dies within ~15 Airy lengths — 30× overhead measured** — `lumenairy/elements/_lens_traced_uniform.py:928-935`

**What is wrong.** `dark = rgrid > r_c` selects the whole grid outside the caustic ring, and `_fold_airy_eval` (scipy `airy` + a complex `exp` over the full set) runs on all of it. The tail amplitude is `Ai((r - r_c)/l_airy)`, which is 6e-19 at 15 `l_airy` and is *clamped to numerical zero* by `_AIRY_ARG_CAP = 50` anyway (`:933-934`) — so ~all of that work produces values the caller cannot distinguish from 0.

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/final_batch.py` §F, λ=1 µm plano-convex f/2, `output_plane_distance = 1.9 mm`, `ray_subsample=4`:

```
N=1024 dx=2.8 um : multibranch 0.08 s, uniform 0.25 s   (fold under-resolved -> fell back)
N=2048 dx=1.4 um : multibranch 0.85 s, uniform 25.42 s (+24.57 s)
                   dark pixels filled = 4 193 535 of 4 194 304 (100.0%)
                   r_c = 21.9 um, l_airy = 3.028 um
```

and at N=4096/dx=0.35 µm (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/fold_rs2.py`): multibranch 11.9 s, uniform **93.1 s**. The physically non-zero annulus is `r_c < r < r_c + 15·l_airy` = 21.9…67 µm, i.e. ~0.6 % of the pixels at N=2048.

**Impact.** ~30× wall-clock on the uniform path at N=2048, ~8× at N=4096, entirely avoidable.

**Fix.** `dark = (rgrid > r_c) & (rgrid < r_c + _AIRY_TAIL_CELLS * l_airy)` with `_AIRY_TAIL_CELLS ≈ 20`, leaving `E_mb` (already zero there) outside. Keep the existing `zfloor` clamp as a belt-and-braces guard. Same change makes the `E_out = E_mb.astype(np.complex128, copy=True)` full-grid copy the only remaining O(N²) cost.

---

### **[P2] The triangle rasteriser materialises ~15–18 full `(n_tri_in_bucket, 2^c, 2^c)` float64 temporaries per bucket — 12.2 GB RSS and 15 minutes measured for ONE default-argument call at N=4096** — `lumenairy/elements/_lens_traced_multibranch.py:596-676`

**What is wrong.** Triangles are bucketed by `cls = ceil(log2(bounding-box width))` and each bucket is processed as one batch: `gx`, `gy`, `PX`, `PY`, `wx`, `wy`, `a0`, `a1`, `a2`, `vmask`, `inside`, the `T` intrapolation expression (several full-size temporaries alive at once) and `Ein_tri` (complex128) are all `(n_s, 2^c, 2^c)`. Nothing bounds `n_s × 2^{2c}` — there is no analogue of `_lens_imap`'s `_IMAP_FIT_CHUNK_ENTRIES` / `_IMAP_EVAL_CHUNK` row-blocking, which the sibling module added for precisely this failure (`FIX_RUNNER_OOM_2026_08_13`).

The worst case is **at the exit vertex** (`output_plane_distance = 0`, the default), where the map is near-identity so every mapped triangle has the same size and they all land in one bucket.

**Evidence (measured).** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/bucket.py` (static bucket census) and `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/mem_time.py` (live):

```
fold case N=4096 sub=4    n_tri=1 187 424  worst bucket c=2 n=561 892  wb=4 -> 0.072 GB/array
fold case N=4096 sub=2    n_tri=4 766 396  worst bucket c=2 n=1 219 826 wb=4 -> 0.156 GB/array
AT the exit vertex N=4096 sub=4  n_tri=1 187 616  worst bucket c=3 n=1 187 616 wb=8 -> 0.608 GB/array
10mm singlet at pupil N=4096 sub=2  n_tri=1 550 656  worst bucket c=2 n=1 550 656 wb=4 -> 0.198 GB/array
```

Live `tracemalloc` on the same calls, all at `output_plane_distance = 0` (the default):

```
N=2048 sub=4 d_out=0    144.04 s   traced peak 1.94 GB   pixels>=2 branches 0   P_out/P_in 0.9998
N=2048 sub=2 d_out=0    190.88 s   traced peak 2.68 GB   pixels>=2 branches 0   P_out/P_in 0.9999
N=4096 sub=4 d_out=0    881.76 s   traced peak 7.74 GB   pixels>=2 branches 0   P_out/P_in 1.0000
```

`tracemalloc` counts Python-traced allocations only; `tasklist` during the N=4096 run showed **12 178 272 K = 12.2 GB** process RSS, and that run drove this workstation into swap for several minutes (plain `cat` of a 0-byte file timed out repeatedly while it was live). The transient scales as `(launch_radius/dx)²` — independent of `ray_subsample` — so N=8192 projects to ≈31 GB traced / ≈49 GB RSS.

**The wall-clock is the bigger surprise and has the same root cause.** `output_plane_distance = 0` is the *worst* case for this rasteriser: the map is near-identity, so every triangle gets `wb = 8` (64 candidate pixels) of which ~1 is actually inside — ~98 % of the barycentric arithmetic is discarded. The same N=2048 grid at `output_plane_distance = 1.9 mm` (near focus, triangles compressed into `wb = 4` buckets) runs in **0.85 s** against **144 s** at the exit vertex: a **170×** penalty for the default argument. `P_out/P_in` is 0.9998–1.0000 there, so energy conservation away from a caustic is excellent — the cost buys nothing.

**Impact.** A routine `apply_real_lens_traced(caustic='multibranch')` at N=4096 with default arguments needs >12 GB of transient and ~15 minutes. `lumenairy.memory.estimate_lens_memory` does not model it (the multibranch is not in that model), so the user gets an OOM kill with no warning — the same failure mode `_lens_imap` already fixed for the fit side.

**Fix.** Chunk the `for c in np.unique(cls)` body over `s` so that `len(s_chunk) * 2^{2c}` stays under a named entry budget (mirror `_lens_imap._fit_row_blocks`). This is bit-identical — there is no reduction inside the block; only `np.add.at` accumulates, and it is order-independent for the *set* of contributions (the float summation order can change, which is the same caveat the file already states at :514-520).

---

### **[P2] The Ludwig caustic-band swap is a per-pixel Python loop over every multi-branch pixel** — `lumenairy/elements/_lens_traced_multibranch.py:693-717`

**What is wrong.** After the vectorised rasterisation, the uniform-fold replacement runs

```python
for s, e in zip(starts, ends):
    ...
    o2 = np.argsort(Ts); dT = np.diff(Ts[o2]); jmin = int(np.argmin(dT))
    ...
    E_flat[bi_s[s]] += uni - (plain[ia] + plain[ib])
```

one Python iteration, four small NumPy calls and one scalar `scipy.special.airy` call per *pixel* that carries ≥2 branches. `ludwig_fold` re-imports `scipy.special.airy` on every call (`:115`). On a broad fold ring (`n_branch` mean 1.92 over the covered region in my N=1024 probe) this is O(N²) Python iterations.

**Evidence (measured).** Same call, `caustic_band='plain'` vs `'ludwig'` (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/mem_time.py`), λ=1 µm f/2 singlet at `output_plane_distance = 1.9 mm`:

```
N=2048 sub=2  plain   15.27 s | ludwig  18.50 s  -> +3.22 s over  3 004 multi-branch pixels = 1073 us/pixel
N=4096 sub=2  plain   91.62 s | ludwig 100.77 s  -> +9.14 s over 12 020 multi-branch pixels =  761 us/pixel
```

~0.8–1.1 ms per multi-branch pixel (the figure includes the one-off `np.argsort(bi)` grouping, which is also inside the `ludwig` branch). At N=1024 / dx=20 µm / `d_out=40 mm` the branch-count diagnostic reads `n_branch max 2, mean over nonzero 1.9247` — i.e. essentially the whole illuminated disc is a two-branch region; at ~0.8 ms/pixel a 1.5M-pixel two-branch ring would cost ~20 minutes.

**Physics datum from the same run** (worth recording, not a defect): the Ludwig swap moves 25 % of the reconstructed grid power on this fold — `P_out/P_in` goes 1.1801 (`plain`) → 0.8884 (`ludwig`) at N=2048 and 1.1585 → 0.8873 at N=4096, against 1.0000 for my independent ASM oracle. So the swap does improve the energy balance (|Δ| 0.18 → 0.11) but neither branch sum is energy-conserving at a resolved fold.

**Fix.** The loop is fully vectorisable: group boundaries are already computed (`starts`/`ends`); build a ragged→padded `(n_group, max_branches)` array of `bT`/`bA`, sort along axis 1, take `argmin` of the gaps, mask the groups inside the band, and call `ludwig_fold` once on the selected vectors. `ludwig_fold` is already elementwise. Hoist `from scipy.special import airy` to module scope.

---

### **[P2] The GRAM guard makes the returned FIELD a function of the machine's free RAM, and `report_refusal` tells the user the opposite** — `lumenairy/elements/_lens_imap.py:1510-1526` and `:1816-1825`

**What is wrong.** `_imap_ram_budget()` reads `lumenairy.memory.get_ram_budget()` (psutil's *AVAILABLE*, i.e. a live, non-deterministic quantity). If the projection exceeds half of it, `build_inverse_map` returns `None`, `apply_real_lens_traced` keeps the coarse-Newton+`map_coordinates` path, and the returned field is a **different field**: the module's own flag docstring (`:118-141`) records that turning the map on moves design 121's banner from FWHM 3.350 → 3.450 µm and changes the peak by 0.8 %, and that the incumbent carries 6.0e-3–1.08e-2 waves rms of core WFE against the exact-trace oracle.

Yet `report_refusal` tells the user: *"The call KEEPS the shipped coarse-Newton + map_coordinates upsample path unchanged (refuse, never degrade), so **this costs speed, never accuracy**."* That directly contradicts the module's own measurement three hundred lines above.

Reachability: `_need_b = 4·n_good·P·8 + …`; the `test_niche_c1_consolidation` exit-NA case reaches `n_good = 5 764 801` at P = 120 → 22.1 GB projected. A 128 GB box builds; a 32 GB box with 16 GB free refuses. Same call, same inputs, different physics.

**Impact.** Non-reproducible results across machines and across runs on the same machine, with a message that actively discourages the user from investigating.

**Fix.** (a) Correct the `report_refusal` wording to say what a refusal costs, citing the module's own numbers. (b) Make the refusal *deterministic* by default: price against a fixed, explicitly-set budget (`lumenairy.set_max_ram`) and treat "psutil AVAILABLE" as a diagnostic rather than a gate, or at minimum include the resolved budget in the guard record and the provenance banner so a run can be reproduced.

---

### **[P3] `_count_interior_turning_points` — the robust turning-point counter — is dead code, and the live traces use the fragile construction it was written to replace** — `lumenairy/elements/_lens_traced_uniform.py:109-127` vs `:237-239`, `:600-601`

The helper's own docstring says it is "robust to an isolated zero-slope SAMPLE at an extremum (which would otherwise double-count a `+ -> 0 -> -` transition)". It is defined at line 109 and **never called** (verified: the only occurrence of the identifier in the file is its `def`). `_trace_meridional_fold` (:237-239) and `_trace_meridional_cusp` (:600-601) both use the raw `np.where(np.diff(np.sign(np.diff(xo))) != 0)` form. A single exactly-zero slope sample at the fold makes `sign` read `[+1, 0, -1]`, `diff` gives two non-zero entries, `n_turn` reads 2, and the call is misrouted from the fold-Airy path to the Pearcey cusp path (`:829`). **Fix:** call the helper, or delete it and stop claiming the robustness in a docstring.

### **[P3] `_build_pearcey_cusp_field(E_mb, geom, wavelength, dx)` never uses `wavelength`** — `lumenairy/elements/_lens_traced_uniform.py:649`

Verified by inspection of the whole body. Either the Pearcey control map is genuinely λ-free (in which case drop the parameter and say so) or a `k`-scaling is missing from the envelope. Given that `_solve_pearcey_control` carries "the scale … in radians so the Pearcey `k`-scaling is implicit" (`:320`), the former is likely — but the dead argument is exactly the shape of a missing term and should not be left ambiguous.

### **[P3] `L0` / `M0` are rebound mid-function from launch direction cosines to per-vertex slowness components** — `lumenairy/elements/_lens_traced_multibranch.py:443-444` vs `:576-581`

`L0 = kcx / k0` (the input congruence tilt) is silently overwritten by `L0 = pn * _g(L[V0i, V0j])` (vertex-0 eikonal-gradient x-component) inside `if good.any():`. Harmless today only because nothing reads the launch cosines after the `_trace_launch_grid` call. Rename to `p0x`/`p0y`.

### **[P3] `_lens_jax` launches rays 2 % OUTSIDE the aperture while its comment says "just inside"** — `lumenairy/elements/_lens_jax.py:508-518`, `:800-810`

`launch_radius = 0.5 * float(aperture) * 1.02` under the comment "Stay just inside the physical aperture". The sibling modules use `* 0.98` with the opposite rationale ("stay strictly inside the aperture rim: rays AT the rim can die … and NaN-poison the finite-difference Jacobians", `_lens_traced_multibranch.py:460-465`). Rays at 1.02·r_ap can miss the surface caps, die, and be dropped by the fit's NaN weighting — a silently smaller fit domain. Pick one policy across the family and name the constant.

### **[P3] The KMAH NaN-fill uses `np.roll`, so a bad node on one edge of the launch lattice can inherit the Maslov index from the opposite edge** — `lumenairy/elements/_lens_traced_multibranch.py:317-327`

`src_m = np.roll(m, sh, axis=ax)` wraps. On a vignetted rim the "nearest valid neighbour" can be 2·launch_radius away. Use edge-clamped shifts (slice assignment) instead of `np.roll`.

### **[P3] Import-time filesystem probe and unsynchronised lazy init in `_lens_imap`** — `:410`, `:417-429`, `:432-488`

`_NUMBA_AVAILABLE = _ilu.find_spec('numba') is not None` runs at module import (measured 38 ms self time for the module on this box, most of it that probe). `_load_numba` / `_get_imap_eval_numba` then mutate the module globals `_numba`, `_njit`, `_prange`, `_NUMBA_KERNELS` with no lock, while the module is otherwise scrupulous about locking (`_IMAP_LOCK` has a 40-line comment on granularity). Two threads reaching `eval_into` on a cold kernel both compile, and `@njit(cache=True)` writes the same on-disk cache entry concurrently. Move the probe into `_load_numba` and guard the lazy init.

### **[P3] The multibranch energy tripwire is one-sided: it can only see energy GAIN** — `lumenairy/elements/_lens_traced_multibranch.py:733-751`

`_ENERGY_BLOWUP_FACTOR = 10.0` fires on `p_out > 10 · p_in`. I measured a 11.3 % energy **loss** on a resolved fold (`P_out/P_in = 0.887` at N=4096; `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/fold_rs2.py`), from the skipped degenerate triangles plus the missing dark-side tail — invisible to the tripwire. Add a lower bound (the same probe, `p_out < 0.5 p_in`), which is cheap and catches under-covered rasterisation.

---

## Performance opportunities (estimated gain; how measured/estimated)

| # | Where | Gain | Basis |
|---|---|---|---|
| 1 | `_lens_traced_uniform.py:928-935` — restrict the dark fill to `r_c < r < r_c + 20·l_airy` | **~30×** on the uniform path at N=2048 (25.4 s → ~1.5 s); ~8× at N=4096 (93.1 s → ~13 s) | measured, `final_batch.py` §F + the 0.6 % physically-non-zero pixel fraction |
| 2 | `_lens_traced_multibranch.py:596-676` — chunk the bucket batch | 12.2 GB RSS / 7.74 GB traced → a named budget (e.g. 1 GB) with no arithmetic change | `tracemalloc` + RSS; `bucket.py` census |
| 2b | same block — skip candidate pixels outside the triangle's *exact* bounding box instead of the power-of-two one | at `output_plane_distance = 0` (the default) ~98 % of the barycentric arithmetic is on candidates that fail `inside`; **144 s (N=2048) / 882 s (N=4096) per call** vs 0.85 s for the same grid near focus | measured, `mem_time.py` |
| 3 | `_lens_traced_multibranch.py:693-717` — vectorise the Ludwig pair-swap; hoist the `scipy.special.airy` import out of `ludwig_fold` | removes 0.76–1.07 ms **per multi-branch pixel** (+3.22 s / 3004 px at N=2048, +9.14 s / 12 020 px at N=4096); a 1.5M-pixel two-branch ring would cost ~20 min | measured `plain` vs `ludwig` A/B, `mem_time.py` |
| 4 | `_lens_traced_uniform.py:773-778` — the uniform path runs the *whole* multibranch then throws the dark half away | the multibranch's rasterisation could be restricted to `r <= r_c + margin` once the fold geometry is known; the meridional trace (`n_fan=4000`, ~ms) is 4 orders cheaper than the 2-D rasterisation and could run **first** | 0.85 s (N=2048) / 11.9 s (N=4096) of multibranch per uniform call, measured |
| 5 | `_lens_jax` — `jit` on the default path (finding P1 #3) | compile is 2.9 s, steady-state 1.08 s/call vs 1.04 s/call eager at N=128–192 → **jit buys ~nothing here**; the real cost is the `(cheb_order+1, N, N)` stacks built twice inside `_cheb_eval_value_grad_jax` per Newton iteration | measured, `jax_grad.py` / `jax_radii.py` |
| 6 | `_lens_jax._cheb_eval_value_grad_jax` | `Tu[K1]`, `Tv[K2]`, `Tup[K1]`, `Tvp[K2]` each materialise `(n_terms, N, N)` = 66×N² float64 — 8.6 GB at N=4096, ×4 arrays, ×2 calls per Newton iteration, ×`newton_iters`, all retained on the autodiff tape. A `jnp.einsum` over the two `(order+1, N, N)` stacks with a precomputed `(n_terms, order+1)` selection matrix, or `lax.scan` over terms, removes the gather entirely | static count; the module's own Notes (`:424-435`) acknowledge the monolithic allocation but not the ×4 gather |

`InverseCharacteristic.eval_into` measured **145.3 ns/pt for 3 channels** (48.4 ns/pt/channel, numba kernel confirmed live) against `scipy.ndimage.map_coordinates(order=3)` at **118.6 ns/pt for 1 channel** on 2e6 points, 20 cores — a **2.45×** per-channel win, consistent with the 2.1× the module's header claims (1.910 s vs 4.035 s). The *absolute* numbers in that header did not reproduce here (my 3-channel rate extrapolates to 9.7 s at 6.71e7 points vs the claimed 1.910 s); the ratio is the reproducible part and the claim should be restated as a ratio.

---

## Alternative algorithms / methods

1. **Replace the branch-enumeration multibranch with a ray-to-wave (Kirchhoff/ASM) hand-off from the exit pupil.** Sample the exit-vertex-plane field from the traced rays (amplitude `|E_in|·√(ρ dρ / r dr)`, phase `k·OPL`) — where geometric optics is *exact*, far from any caustic — then propagate with a band-limited angular spectrum. This is Zemax POP's / CODE V BSP's hand-off. It is exact through folds, cusps, the axial point focus and the D5 catastrophe alike, needs no KMAH bookkeeping, no `1/√|J|`, no Ludwig swap and no dark-side completion. **I wrote exactly this as the oracle for this audit** (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/fold_rs2.py`, ~25 lines of ASM): at N=4096 it took **6.1 s** against the multibranch's **11.9 s** and the uniform's **93.1 s**, and it reproduced the fold's dark-side tail the multibranch drops entirely (I_oracle 610/259/140/31 at r = 22.05/23.10/24.15/25.20 µm where the multibranch is identically 0). References: Goodman, *Introduction to Fourier Optics* 3rd ed. §3.10; Matsushima & Shimobaba, *Opt. Express* **17**, 19662 (2009) (band-limited ASM). Cost and validity: needs the pupil sampled at ≤ λ/(2 NA) and the grid to satisfy `z·tanθ_max < N dx/2` — both trivially met at the pupil, where the multibranch already needs a fine grid anyway.

2. **Gaussian-beam decomposition (`propagators/gbd.py`) instead of the multibranch for the D5 axial catastrophe.** The Ludwig swap regularises only the *closest pair*, so at a rotationally-symmetric on-axis focus the residual ring branches keep divergent ART amplitudes — I reproduced this at 2.12e+04× the input power with 1289 branches on one pixel (`repro_uniform_overwrite.py`). GBD has no caustic singularity by construction (no KMAH index, no `1/√|J|`). Popov, *Wave Motion* **4**, 85 (1982); Červený, Popov & Pšenčík, *Geophys. J. R. astr. Soc.* **70**, 109 (1982); Hill, *Geophysics* **55**, 1416 (1990). The module already routes users there in prose (`:746-751`) — it could route there automatically when the energy tripwire fires.

3. **Complex rays instead of the bright-side least-squares fit for the dark tail.** `apply_real_lens_traced_uniform` currently *fits* `(c0, c1)` to the multibranch bright field over a ~1·`l_airy` band and refuses when the residual exceeds 0.5 (`:918-923`). The CFU coefficients are analytic in `ζ`, so they can be obtained directly by continuing the two coalescing saddles into complex ray space (Chapman & Drummond, *Bull. Seismol. Soc. Am.* **72**, S277 (1982); Kravtsov & Orlov, *Caustics, Catastrophes and Wave Fields*, Springer 1999, ch. 5). That removes the fit, the fit-band width heuristic, the `1/√r` weighting and the `fit_residual > 0.5` fallback in one step, and it works where the bright side is under-resolved (the `l_airy < 1.2 dx` refusal at `:889`).

4. **Exact Pearcey derivatives instead of finite differences.** `_pearcey_basis` (`:430-440`) takes `∂P/∂x`, `∂P/∂y` by central differences with a hard-coded `step=3e-3` on clamped control coordinates. `P` satisfies closed recurrences that give both derivatives in terms of `P` itself at machine precision and at no extra cost: Connor & Curtis, *J. Phys. A* **15**, 1179 (1982); Kirk, Connor & Curtis, *J. Phys. A* **33**, 4797 (2000). This also removes the arbitrary step, which currently sets an error floor of O(step²·P''') on the cusp amplitude.

5. **Maslov (phase-space) propagator for the fold itself.** `lenses_maslov.apply_real_lens_maslov` already implements the mixed-representation integral; a fold that is single-valued in `(x, p_y)` needs no branch enumeration at all. Maslov & Fedoriuk, *Semi-Classical Approximation in Quantum Mechanics* (Reidel, 1981); Klimeš, *Stud. Geophys. Geod.* **54**, 269 (2010) App. B (the same reference the multibranch cites for its KMAH sign). Useful as the *dispatch target* when `_trace_meridional_fold` refuses rather than falling back to a bright-side-only field.

6. **For `_lens_imap`'s Chebyshev evaluation:** Clenshaw recurrence in the *2-D* form (evaluate in `v` first with `u`-Clenshaw coefficients) replaces the `M`-term inner loop with `O(degree)` work per pixel — `120` MACs/pixel/channel → `2·15` — at the same accuracy. Clenshaw, *MTAC* **9**, 118 (1955). At the measured 48 ns/pt/channel this is the difference between the kernel being memory-bound and compute-bound.

---

## Code organization observations

* **The five files re-implement the same scaffolding four to six times.** Measured occurrence counts across `_lens_traced.py` / `_lens_traced_multibranch.py` / `_lens_traced_uniform.py` / `_lens_imap.py` / `_lens_jax.py` / `_lens_real.py`:

  | pattern | LT | MB | UNI | IMAP | JAX | REAL |
  |---|---|---|---|---|---|---|
  | exit-vertex correction (`t_to_vertex`) | 8 | **0** | **0** | 0 | 8 | 0 |
  | `launch_radius = 0.5·aperture·f` | 0 | 1 (×0.98) | 1 (×0.98) | 0 | 2 (×1.02) | 0 |
  | `n_launch` odd-parity bump | 1 | 1 | 0 | 0 | 2 | 0 |
  | `x = (arange(N) - N/2)·dx` | 4 | 2 | 4 | 0 | 2 (jnp) | 0 |
  | `N_z` 1e-30 guard | 4 | 3 | 4 | 0 | 0 | 1 |
  | mirror-in-surfaces guard (15-line raise) | 1 | 0 | 0 | 0 | 2 | 1 |
  | Chebyshev machinery | 15 | 0 | 0 | 8 | 3 | 0 |

  The zero in the `t_to_vertex` row **is** the P0. A single `lumenairy/elements/_traced_common.py` carrying `advance_to_exit_vertex(ex, surfaces, wavelength)`, `resolve_launch_lattice(prescription, N, dx, sub) -> (launch_radius, n_launch, xs_in)`, `grid_axis(N, dx)` and `guard_no_mirrors(prescription, caller)` would have made this defect impossible to introduce, and would remove ~150 duplicated lines.

  Note what is **not** duplicated: none of the five files carries a cupy or numexpr scaffold (27 and 4 hits in `_lens_traced.py`, 12 and 39 in `_lens_real.py`, **0** in all five of mine). That part of the brief's hypothesis does not hold.

* **Three independent Chebyshev implementations** (`_lens_traced._Cheb2DEvaluator` / `_cheb2d_val_grad_numba`, `_lens_imap._td_design`/`_td_design_grad`/`_imap_eval_numba`, `_lens_jax._cheb_T_recurrence_jax`/`_cheb_Tprime_recurrence_jax`) with three different total-degree index orderings built three different ways (`_td_terms`, `_cheb_total_degree_indices`, and `_lens_traced`'s own). `_lens_imap._td_terms`' docstring asserts it matches `_lens_traced`'s ordering; nothing tests that `_lens_jax._cheb_total_degree_indices` does.

* **Comment-to-code ratio.** `_lens_imap.py` is 1841 lines of which the flag block `:101-405` is ~300 lines of prose defining 13 constants, and `_traced_flags.py` is 476 lines defining a 41-entry table with ~180 lines of narrative. The narrative is genuinely load-bearing (it records measurements), but it is in the wrong place: `_lens_imap`'s `TRACED_INVERSE_MAP` docstring (`:104-196`) is 92 lines and contains the one sentence that matters for a caller ("ships True, refuse-never-degrade") buried at line 109. This material belongs in `docs/audits/`, with the constants carrying one-line references.

* **Docs that contradict behaviour** (each already listed as a finding): `_lens_traced_multibranch`'s "past the prescription's exit vertex" (P0); `_lens_jax`'s "JAX's vmap+JIT replaces the first" (P1); `_lens_jax`'s "Stay just inside the physical aperture" over `* 1.02` (P3); `_lens_imap.report_refusal`'s "costs speed, never accuracy" (P2); `_lens_traced_uniform._count_interior_turning_points`'s robustness claim for a function nothing calls (P3).

* **`_traced_flags.py` is well built and I found no defect in it.** `traced_flags` / `traced_era` capture-then-restore correctly (including the nested case: `traced_era` applies the preset, then `traced_flags` saves the *post-preset* values, and the two `finally` blocks unwind in the right order). `_LOCK` is an `RLock`, so the nesting cannot deadlock. It is held across the `yield`, which serialises concurrent blocks — documented at `:353-358` as the intent. My only substantive observation is that the registry is the *right* place to have caught the P2 cache defect: `DETERMINISTIC_TRACED_FIT`'s own entry (`:253-261`) states the bit-for-bit contract that the imap cache silently breaks, and nothing cross-checks the two.

---

## Unverified suspicions

* **The Pearcey cusp path is unverified end-to-end.** I could not build a prescription that reaches `_trace_meridional_cusp` with `ok=True` in the time available (my `n_turn == 2` attempt refused at `bad_kappa` because the same plane carries the on-axis catastrophe). Specifically unverified: the sorted-phase→sorted-critical-value pairing in `_solve_pearcey_control` (`:383-388`) is only valid if the two orderings are *monotonically related*, which is asserted nowhere; and `_pearcey_cusp_amp_coeffs` pairs branches by `argsort(phis)` against roots by `argsort(cv)` (`:413-415`), which inherits that assumption. A fixture would be `tests/unit/test_niche_r2_pearcey_cusp.py`'s own prescription.
* **`_lens_traced_uniform`'s dark fill is `rgrid > r_c` unconditionally** (`:929`), where `r_c` is `|x_out|` at the *signed* turning point. If a prescription ever produces live rays landing at `|x_out| > r_c` (an axis re-crossing past the marginal focus), the fill would erase real bright field. I tried to reach it (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/repro_uniform_overwrite.py`) and the fold trace refused first (`bad_kappa`), so on the cases I could build it is unreachable. To confirm or clear: find a prescription where `_trace_meridional_fold` returns `ok=True` *and* `max|x_out| > r_c`.
* **`amplitude='analytic'` under `jax.grad`.** `_amp_callback_jax_linear` wraps `jax.pure_callback` in a `custom_jvp`. Reverse-mode requires JAX to *transpose* the JVP rule, and `pure_callback` has no transpose rule — so `jax.grad` on that leg may raise. Not tested. One `jax.grad(lambda E: jnp.sum(jnp.abs(TJ(E, ..., amplitude='analytic'))**2))` call would settle it.
* **`_incumbent_fingerprint` collapses all refusals to one hash.** Any incumbent that raises `ValueError`/`TypeError`/`ArithmeticError` on the strided probe hashes as `<raised:ValueError>` (`:1048`), so two *different* refusing incumbents share a key. Narrow, and I did not construct a case.

---

## Checked and found correct (brief — so coverage is known)

* **The KMAH sign is right.** Under `exp(-iωt)` / `exp(+ikS)` a fold crossing must multiply the touched branch by `exp(-iπ/2)`, and `ph_br = np.exp(-0.5j*np.pi*m_br)` (`_lens_traced_multibranch.py:566`) does. Verified analytically and numerically against the exact cubic-phase (Airy) integral (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/ludwig_check.py`): the two-branch sum with `-π/2` converges to the exact Airy field as `k^{2/3}ζ` grows (rel. error 2.57e-02 → 1.62e-03 → 1.44e-05 at `k^{2/3}ζ` = 10, 30, 100) while the `+π/2` control stays at 5.60e+00 → 1.94e+00 → 7.14e-01. Consistent with `CONVENTIONS.md` §7 and with the Gaussian-beam Gouy shift (two crossings = `exp(-iπ)`).
* **`ludwig_fold` reproduces the exact CFU/Airy form.** Relative error 0.0 – 8.9e-16 at k = 1e3, 1.1e-14 at k = 1e5, ≤ 8.5e-12 at k = 1e7, over ζ ∈ {2e-3, 1e-2, 5e-2} and three complex `(a0, a1)` pairs. The "machine precision (~3e-16)" claim holds at moderate `k` and degrades gracefully; the formula, including the `ρ^{±1/4}` split and the `1e-300` guard, is right.
* **`lenses_maslov._fold_airy_eval` is the correct CFU kernel.** `∫(a0 + a1 s)exp(ik(s³/3 − ζs))ds = 2π[a0 k^{-1/3}Ai(−k^{2/3}ζ) − i a1 k^{-2/3}Ai′(−k^{2/3}ζ)]` — derived independently and matches the code exactly, including the analytic continuation to ζ < 0.
* **`jax.grad` through `apply_real_lens_traced_jax(radii=...)` is correct.** With a phase-sensitive merit (on-axis far-field / Strehl), `jax.grad = -0.064058599` vs central FD `-0.064059035` at h = 1e-6 → **6.8e-06 relative**. Localised the chain: `d(ΣOPL)/dR` through `trace_jax_with_params` matches FD to 2.3e-09; `d(Σcoeffs)/dR` through `_cheb_fit_2d_jax` (including the NaN masking via double `jnp.where`) matches to 4.2e-08. No `stop_gradient` misuse, no NaN trap. (My first probe reported a zero gradient — that was my merit's fault: at the exit vertex the intensity is phase-independent. Reported here so the false positive is not repeated.)
* **`jax.grad` w.r.t. `E_in` works for both entry points**: finite, `max|g| = 2`, 7409/16384 non-zero (the zeros are the aperture mask).
* **`_lens_jax`'s launch-lattice sizing is adequate at the defaults.** I suspected `ray_subsample=8` + `cheb_order=10` silently starves the fit; measured against my exact ray oracle it does not — OPD rms is 2.47e-10 waves at 13×13 nodes (2.6 samples/term) and 1.91e-10 at 205×205 (637 samples/term). **Suspicion dropped.**
* **The multibranch's fold physics is qualitatively right where it applies.** Against my independent ray-to-wave ASM oracle on a resolved fold (λ=1 µm, f/2, N=4096, dx=0.35 µm): fringe *positions* in the two-branch zone match, bright-side intensity rms error 0.206 of peak (much of it the near-axis Bessoid ringing the 2-D ART cannot represent), and `apply_real_lens_traced_uniform`'s dark-side Airy tail recovers the first ~2 fringes of the tail the multibranch drops to zero (458 vs 610, 250 vs 259, 122 vs 140 at r = 22.05/23.10/24.15 µm vs `r_c` = 21.9 µm). Total-power ratios: 1.028 (pupil-plane disc, N=1024), 0.887 (resolved fold, N=4096), 0.9997 (exit vertex vs the single-valued path), and **0.9998–1.0000 at the exit vertex against the input aperture power** at N=2048/4096 — i.e. the rasteriser conserves energy essentially exactly away from a caustic, and the Ludwig swap improves the balance at a fold (1.18 → 0.89 against the oracle's 1.000).
* **Flags are genuinely read at call time.** An AST-free scan of every `from X import Y` in `lumenairy/` against all 41 registered flag names found exactly three from-imports (`carrier.py:2954`, `carrier.py:8020`, `fga.py:3054`) and **all three are inside function bodies**, so they re-read on every call. No module snapshots a flag at import.
* **`ERAS[-1]` reproduces the live shipped defaults.** Checked all 21 flags that carry era entries against `traced_flag_state()` — every one matches `resolve_era('v5.34')`.
* **No import cycle.** `python -X importtime -c "import lumenairy.elements._lens_imap"` shows `_lens_imap` completing (40 ms cumulative) *before* `_lens_traced` (46 ms cumulative) — `_lens_traced` imports `_lens_imap` at module scope, `_lens_imap` imports `_lens_traced` only inside function bodies. The claim at `_lens_traced.py:30-33` holds.
* **`_lens_traced_uniform._radial_amp_sampler`'s odd-N fix is right.** `c = ceil(N/2)` and `row = argmin(|x|)` give the correct first-non-negative column and nearest-axis row for both parities; `rp = hypot(x[c:], x[row])` is the true pixel radius.
* **The CFU fit weighting is right.** `wt = 1/sqrt(rb)` (`:912`) makes the squared weight `1/r`, which times the `∝ r` pixel count per radius is constant per radius — exactly what the comment claims.
* **`_lens_imap`'s central accuracy claim holds.** On a real 129×129 singlet congruence the degree-14 model beats a spline-Newton incumbent by 5.1e+03× on OPL (3.60e-12 vs 1.84e-08 waves at the off-lattice probe; 1.23e-12 vs 3.99e-09 waves rms at 4000 independent random points) and by 1.1e+03× on entrance position (4.8e-18 vs 5.3e-15 m), building in 0.78 s. The `G8` envelope rule, the R2 probe placement and the `census_amp` restriction all behave as documented.
* **`InverseCharacteristic.eval_into` with a non-contiguous 1-D output array writes through correctly** (I suspected `np.asarray(o).reshape(-1)` would silently copy; for 1-D it is a no-op). Suspicion dropped.
* **`_traced_flags.traced_flags` / `traced_era` save-and-restore semantics** verified by reading: values captured before any write, restored in `finally` in reverse order, nested preset+override unwinds correctly, `RLock` prevents self-deadlock.

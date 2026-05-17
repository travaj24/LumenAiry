# Lumenairy v4.12.0 — Round-5 Pre-PyPI Audit
Date: 2026-05-16
Codebase: ~80,000 LOC, ~80 Python files, +8132 LOC across 39 files in v4.12.0
Method: 13 parallel audit agents covering (a) verification of every v4.12.0 round-4 fix,
(b) correctness audit of v4.12.0 perf changes (pyFFTW, caches, JIT, einsum chunks),
(c) currently-modified WIP files, (d) deferred items, (e) PyPI release readiness, and
(f) benchmarks infrastructure.

---

## TL;DR — HOLD release for ~30 minutes, fix 4 release-blockers, ship.

v4.12.0 successfully closes ~20 round-4 release blockers. The physics correctness work is
genuinely strong: all Tier-0 / Tier-1 / Tier-2 round-4 findings have pinning tests, most with
real ground-truth comparisons (Cassegrain hand-derived `S4 = -16` to 1e-9; C-LR-1 vs
`apply_real_lens_traced`; RS-vs-ASM on-axis phase). Test quality has dramatically improved:
**3% wrong-reason rate, down from 33% (round-3) and 22% (round-4)**. The Tier-1 ~10×
performance speedups are real and correctness-pinned against pre-perf references at 1e-12
relative tolerance.

**However, 4 PyPI-release blockers remain — all ~30-min fixes** but the same brand-damaging
first-impression class that round-4 was supposed to close. Plus 3 documentation/code drift
items (release notes contradict shipping source in three places), 2 untouched modules with
silent data-loss bugs (`io/storage.py` append-side complex64 demotion; `io/codegen.py`
aperture-stop drop), and one freshly regressed item (v4.12.0's new pyFFTW plan cache is not
covered by `clear_asm_caches` / `lumenairy_context(clear_caches_on_exit=True)`).

---

## PyPI release blockers (must fix before tagging)

### A1. `pytest-benchmark` missing from `[project.optional-dependencies].dev`
Release notes (line 70) and CHANGELOG (line 182) claim "`pytest-benchmark` added as a dev
dependency". `pyproject.toml` has `dev = ["pytest>=7.0", "pytest-cov"]` — no `pytest-benchmark`
anywhere. User following `benchmarks/README.md`'s `pytest benchmarks/ --benchmark-only -v`
after `pip install lumenairy[dev]` gets `error: unrecognized arguments: --benchmark-only`.

**Fix:** add `"pytest-benchmark>=4.0"` to the `dev` extra (or a new `bench` extra) in
`pyproject.toml`.

### A2. `benchmarks/conftest.py` adds a `bench` marker that's not declared
`conftest.py:12-16` applies `pytest.mark.bench` to every collected item. `pyproject.toml`'s
`[tool.pytest.ini_options].markers` lists only `unit/integration/regression/slow` — **no
`bench`**. Combined with `--strict-markers` in `addopts` (line 144-149), every benchmark
file errors at collection time. The documented `pytest benchmarks/ --benchmark-only -v`
invocation does not work.

**Fix:** add `"bench: pytest-benchmark micro-perf tests (run with --benchmark-only)"` to the
markers list.

### A3. `set_fft_auto_promote` / `get_fft_auto_promote` / `clear_zernike_basis_cache` documented as user knobs but NOT exported
- Release notes: "`set_fft_auto_promote(False)` disables for startup-sensitive workflows"
  (line 25); "`clear_zernike_basis_cache()` exposed for in-place-mutation escape" (line 27).
- Actual code: these exist in `lumenairy.propagators.propagation` and
  `lumenairy.analysis.core` but are NOT imported into `lumenairy/__init__.py` and do NOT
  appear in `__all__`. `import lumenairy as la; la.set_fft_auto_promote(False)` raises
  `AttributeError`.

**Fix:** add to the existing import blocks in `lumenairy/__init__.py` and to `__all__`.
Three-line change. Also expose `clear_lg_polynomial_cache` (same gap, found by two
independent agents). Also wire all three into `lumenairy_context(clear_caches_on_exit=True)`
(currently only `clear_asm_caches()` is called — see A5).

### A4. Stray top-level files in repo root
`C:tmpasymptotic_v4_11_2.py` (135 KB) and `C:tmphf_v4_11_2.py` (14 KB) — created by a botched
`cp C:\tmp\...` on Windows (`:` is a reserved path char so the path collapsed into the
filename). NOT in the wheel and NOT in the sdist (not listed in `MANIFEST.in`, not under
`lumenairy/`), but they're clutter in the source git tag.

**Fix:** `git rm 'C:tmp*'` before tagging.

### A5. (Tier-0.5) `clear_asm_caches` doesn't clear v4.12.0's new pyFFTW plan cache — **newly regressed in v4.12.0**
The v4.12.0 perf pass added `_PYFFTW_PLAN_CACHE` (the new double-buffer cache). `clear_asm_caches`
was not updated to clear it. **CHANGELOG explicitly claims `lumenairy_context(clear_caches_on_exit=True)`
drops the perf-pass caches — it doesn't.** 256 MB to 1 GB per cached plan-key stays resident.
Long-running notebook sessions cycle through dtypes / thread counts → accumulates indefinitely.

**Fix:** extend `clear_asm_caches()` to also clear `_PYFFTW_PLAN_CACHE`, `_PYFFTW_BAD_SHAPES`,
`_ZERNIKE_BASIS_CACHE`, `_LG_POLYNOMIAL_CACHE`, `_PROPAGATE_SYSTEM_JAX_CACHE`, and the three
phase-retrieval JAX kernel caches. Or split into `clear_propagator_caches()`,
`clear_analysis_caches()`, `clear_jax_caches()` and call all from `clear_caches_on_exit=True`.

---

## Documentation / code drift (3 sites — release notes contradict shipping source)

### D1. B1-10 (half-pixel grid drift) is *actually fixed in v4.12.0* but documentation says it's deferred
Release notes (lines 61-63), CHANGELOG (line 161), and README (line 69) all state B1-10 is
**deferred to v4.12.1**. Reality: every site round-4 named is pixel-centred in HEAD source:
- `hf.py` (labelled v4.12.0): lines 101, 300, 319, 383
- `gbd.py` (labelled v4.12.1 in source comment): lines 121-129
- `mhs.py` (labelled v4.12.1): lines 80, 426
- `subaperture.py` (labelled v4.12.1): line 245
- `optimize/core.py` (labelled v4.12.1): line 188

A full pinning test ships (`tests/unit/test_audit_fixes_v4_12_1_grid_unify.py`) — 16 tests
pin central-pixel `axis[N/2] == 0.0` at every site plus a cross-method ASM vs GBD centroid
agreement test. Test is part of the default `pytest tests/` run and contributes to the
390-test claim.

**Fix:** either
- (a) amend release notes / CHANGELOG / README to move B1-10 from "Deferred" to "Round-4
  Tier-1 fixes" and rename the test file to drop the `v4_12_1` suffix; OR
- (b) revert the gbd/mhs/subaperture/optimize changes to genuinely defer (the working-tree
  WIP earlier was for these files — they got committed into v4.12.0 either intentionally or
  via a tagging error).

The audit can't tell which is intended; the user should choose. Either option is fine — but
the current state is internally inconsistent.

### D2. Release notes say Newton spherical fast-path was *reverted* — actually only the analytic-normal stash was reverted
Release notes line 61 says "Raytrace Newton spherical fast-path + analytic-normal stash"
was reverted. Source at `raytrace/core.py:478-525` shows the fast-path **IS shipped** (the
`is_pure_spherical` branch with the ray-sphere quadratic). What was actually reverted is only
the paired analytic-normal switch (cross-backend Maslov drift 1.17e-3).

**Fix:** clarify release notes to "Newton-skip fast-path shipped; the paired analytic-normal
switch was reverted to avoid Maslov asymptotic cross-backend drift."

### D3. Release notes claim `through_focus_scan` NumPy got an H-hoist — it didn't
Release notes line 26: "**`through_focus_scan` (NumPy) H-hoist** — input FFT, kx/ky,
propagating mask, target-dtype all hoisted outside the z-loop. Bit-near-exact (abs err 0.0)
vs per-z reference."

Reading `analysis/through_focus.py:299-307`: NumPy `through_focus_scan` calls
`angular_spectrum_propagate(E_exit, z, ...)` once per z value. **No hoisting.** The JAX
version (`_asm_one_z` line 893) IS hoisted. The 4.7× speedup is real, but comes from
- pyFFTW double-buffer plan cache + auto-promote (the actual v4.12.0 work)
- pre-existing `_FREQ_GRID_CACHE` (z-invariant entries hit across the z-scan)

The pinning test `test_metrics_match_per_z_reference_to_1e_12` is a tautology — same call
path on both sides. **The "bit-near-exact" claim is trivially true because there is no
algorithm change in the NumPy path.**

This was independently caught by the pyFFTW + through_focus agent AND the deferred + cross-
cutting agent.

**Fix:** rewrite the through_focus section of the release notes to: "NumPy
`through_focus_scan` inherits the pyFFTW plan-cache + auto-promote speedups via the underlying
`angular_spectrum_propagate`. The JAX twin (`through_focus_scan_jax`) does hoist the input
FFT and kx/ky outside the vmap (separate work — see release-note item for `through_focus_scan_jax`)."

### D4. Release notes claim a JIT cache exists at the inner ASM kernel inside `through_focus_scan_jax` — none exists
Release notes line 28: "JAX jit caches at ... the inner ASM kernel inside
`through_focus_scan_jax`". Code at `analysis/through_focus.py:895-913`: no `@jax.jit` wraps
the inner kernel, no module-scope cache. Just `jax.vmap(_asm_one_z)(z_jax)` with closures
rebuilding `kz_safe`, `propagating`, `E_fft_shifted` per call → **each Python call retraces**.

The pinning test `test_through_focus_scan_jax_runs` is a smoke test only (`isfinite()` on
`peak_I.shape`); its docstring even acknowledges it can't inspect a public cache.

**Fix:** either implement the cache (jit-wrap the vmapped kernel, module-scope OrderedDict
cache) or correct the release notes + test docstring.

---

## Silent data-loss bugs in untouched modules (3 sites)

### S1. `io/storage.py` `append_plane_h5` and `_zarr_append_plane` hardcode `complex128`
`storage.py:342` and `storage.py:653` use `np.asarray(field, dtype=np.complex128)`. The
round-4 F12 fix patched only the **single-shot** save APIs (`save_field_h5`,
`save_planes_h5`); the **append-side** APIs were missed. Production pipelines using
`MhsPipeline.run(store=...)` or `replay_run` go through the append path. Complex64
simulations silently double their on-disk size and CPU time on every per-step write.

`save_jones_field_h5` (`:282-283`) has the same issue — no `preserve_dtype` exposed.

### S2. `io/codegen.py` silently drops aperture stops from Zemax prescriptions
Docstring at line 275 promises `type='aperture'` step emission and the generation paths handle
it (lines 616-623, 775-778) — but `_decompose_prescription` **never emits one**. Zemax
prescriptions with `STOP` or zero-thickness aperture surfaces will lose the stop in the
generated script → wrong simulation downstream. **High-traffic codepath** for the
"import-from-Zemax" workflow that the README cookbook advertises.

Plus `io/codegen.py` silently defaults `wavelength = 1.31e-6` (1310 nm) with no warning when
neither user nor prescription supplies one. Users converting visible-band Zemax files get NIR
scripts.

### S3. `analysis/ghost.py` module docstring contradicts itself
`ghost.py:35-37` describes `focus_z_estimate` as "harmonic mean of `|R_i|` and `|R_j|`" — but
`R_i` / `R_j` everywhere ELSE in the module refer to **Fresnel reflectance**. The actual code
at `ghost.py:152-154` uses curvature radii. Two different conventions for the same letter in
the same module.

---

## Latent / structural concerns (8 sites)

### L1. `_PROPAGATE_SYSTEM_JAX_CACHE` is unbounded Dict; will leak under iterative optimization
`system.py:495` — plain `Dict`, no eviction, no cap, no public `clear_*` function. Key is
`(sigs_tuple, wavelength, dx, dy)` where `sigs_tuple` carries floats. An optimizer that
varies any of `z`, `f`, `xc`, `yc`, `wavelength`, `dx`, `dy` per iteration creates a new
cache entry per iteration → **1000-iter Newton loop accumulates 1000 compiled XLA binaries.**

Release notes claim "Module-scope **OrderedDict** caches" — these are plain `dict`. Fix:
convert to OrderedDict + `_MAX_CACHE_SIZE = 32` LRU eviction; add `clear_jax_caches()`
public function; wire into `clear_caches_on_exit=True`.

### L2. `set_default_complex_dtype` silently ignored on JAX path
`system.py:833` hard-casts `jnp.asarray(E_in, dtype=jnp.complex64)`. `phase_retrieval.py:600`
does `.astype(jnp.complex64)`. User running `set_default_complex_dtype(np.complex128)` then
calling a JAX path gets float32-precision answers with no warning. JIT cache key does NOT
include dtype, so cross-dtype calls hit the same cached kernel.

`apply_real_lens_traced_jax` (`_lens_jax.py:237-241`) reads `jax.config.jax_enable_x64`, not
`get_default_complex_dtype()`. Inconsistent across modules.

### L3. `PropagationResult` has no `dy` field — anamorphic Fresnel info-loss
`_coerce_field` correctly extracts `dx_out` from tuple-returning kernels but **silently
discards `dy_out`**. `Source` (`sources/core.py:941-967`) also has only `dx`. For anamorphic
Fresnel where the kernel returns distinct `dx_out, dy_out`, the wrapped result loses the
y-axis pitch information silently. For square inputs (`dx == dy`), benign. **Tier-1
silent-information-loss for anamorphic users.**

### L4. Sibling-gap pattern (the round-4 recurring anti-pattern) recurs in v4.12.0
- **`apply_real_lens_traced` mirror guard NOT applied to `apply_real_lens_maslov`,
  `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`** (Tier-2 verification agent).
  Hand-built prescription with `surfaces[i]['is_mirror']=True` slips past on all three
  siblings.
- **`error_reduction(backend='jax')` and `hybrid_input_output(backend='jax')` dispatch don't
  forward `initial_guess`** (Tier-0 verification agent). Tier-0 B2-6 fix landed in
  `gerchberg_saxton` only; sibling dispatchers were missed.
- **`gerchberg_saxton(backend='jax', return_history=True)` silently drops `return_history`**
  — JAX twin returns a 2-tuple, user expecting 3-tuple silently gets wrong shape.
- **`hf.py` has the v4.12.0 grid-convention fix; `gbd.py / mhs.py / subaperture.py / optimize/core.py`
  got the same fix labelled v4.12.1** (the D1 documentation drift above).

### L5. 346 `except Exception:` clauses remain in core scientific code
Round-4 F14 was about bare `except:`. Only 2 true bare-except remain. But 346
`except Exception:` clauses persist across the package, many of them `pass`/return-NaN without
logging. Effect-equivalent silent-failure pattern; needs a sweep.

### L6. `apply_mirror` doesn't use `_xp_of(E_in)`
Lone holdout — every other `apply_*` in `elements.py` switched to array-namespace dispatch
in 4.10/4.11. `apply_mirror` still uses `np.exp`, `np.where` directly → fails silently on
CuPy/JAX fields. Also still missing `dy` (round-3 finding).

### L7. The benchmark for `through_focus_scan_jax` "cold call" pollutes itself
`test_bench_jax_jit.py:through_focus_scan_jax_first_vs_warm` doesn't clear inner ASM JIT
cache (because there isn't one — D4 above). First-call timing is meaningless and execution-order-
dependent.

### L8. `_open_zarr_group_safe` non-thread-safe monkey-patches `Path.mkdir`
`storage.py:582-645` — two threads racing through `append_plane_h5` can leave the patch in
an inconsistent state. Document or guard with a `threading.Lock`.

---

## Round-5 wins (what v4.12.0 got right)

### Test quality — dramatic improvement
- **3% wrong-reason test rate** (4 of ~125 tests in 9 new files), down from 33% (round-3)
  and 22% (round-4).
- **18 of ~20 round-4 fixes have pinning tests** — most with real ground-truth comparison
  (Cassegrain hand-derived S4=-16 to 1e-9; C-LR-1 vs `apply_real_lens_traced`; RS-vs-ASM
  on-axis phase agreement to λ/100).
- Test isolation discipline mostly clean (autouse fixtures for FFT plan cache, per-test
  cache clears for Zernike cache, uncommon `n_iter` values to avoid JAX-cache collision).
- Optional-dep gating clean (no bare `except: return True, 'skipped'` patterns in the new
  files).

### Cross-audit convergence (independent confirmations)
- **`test_band_limit_uses_strict_less_than_numpy` is weak** — Tier-1 agent AND test-quality
  agent both flagged independently.
- **NumPy through_focus_scan has no real H-hoist** — pyFFTW+through_focus agent AND
  deferred+cross-cutting agent both noticed.
- **B1-10 is shipped in v4.12.0 source but documented as deferred** — WIP audit, deferred+cross-
  cutting, AND PyPI release readiness agents all converged.

### Round-4 closures verified
All Tier-0 (README cookbook, deprecation shims, GS dispatcher) and Tier-1 (10 dispatcher/JAX/
RS/SAS items) and Tier-2 (4 items) round-4 findings have **passing pinning tests in HEAD**.
Tier-0 README sweep covered all 22 fenced code blocks; only the original 11 offenders had
issues, all fixed; no other broken examples found. Deprecation shims emit correct
`DeprecationWarning` end-to-end with strong forwarding tests (real `.zmx` file round-trip).

### Sign convention end-to-end consistent
Walked source factory → lens → propagator → analysis. All forward-propagation kernels use
`exp(+i·kz·z)`; all phase screens use `exp(-i·k·OPD)`; round-trip identity preserved. The
v4.12.0 perf changes did not perturb any sign convention.

### Cleanly deferred items (Section A of cross-cutting)
- Raytrace Newton spherical fast-path: ✅ analytic-normal stash reverted; rest of fast-path
  shipped (see D2).
- `trace_jax` flat-tuple jit cache: ✅ no `@jax.jit`, no module cache, test file documents
  the deferral.
- B1-10 half-pixel grid drift: ⚠️ shipped in v4.12.0 source but release notes say deferred
  (see D1).

### Perf changes correctness-pinned
- pyFFTW plan cache + auto-promote: bit-near-exact round-trip (~1e-10 vs reference).
- Zernike basis fingerprint cache: `B1 is B2` identity check; LRU eviction tested.
- LG polynomial `lru_cache`: chunk-size invariance + `is_not` mutation safety, all round-3
  fixes preserved.
- HF Chebyshev chunk vectorisation: 4 chunk-size pins (1/4/16/32) vs hand-rolled reference
  at 1e-10. **The latent shape-mismatch bug claim is fully substantiated** — v4.11.2 author
  left a paper trail (`test_audit_fixes_v4_11_2_hfpi_hf.py:384-387`) that they could not
  actually call the function end-to-end. v4.12.0 chunk-vectorisation closes the underlying
  bug.
- JAX JIT caches: cache-hit cold-vs-warm equality pinned (modulo the missing through_focus
  one — D4).

---

## Recommendations

### Tier 0 — Pre-PyPI tag (~30 min)
1. **A1** — Add `pytest-benchmark>=4.0` to `pyproject.toml` `[project.optional-dependencies].dev`.
2. **A2** — Add `bench` to `pyproject.toml` `markers`.
3. **A3** — Add `set_fft_auto_promote`, `get_fft_auto_promote`, `clear_zernike_basis_cache`,
   `clear_lg_polynomial_cache` to `lumenairy/__init__.py` imports and `__all__`. Wire them
   into `lumenairy_context(clear_caches_on_exit=True)`.
4. **A4** — `git rm 'C:tmpasymptotic_v4_11_2.py' 'C:tmphf_v4_11_2.py'`.
5. **A5** — Extend `clear_asm_caches` to clear the new pyFFTW plan cache and the new perf
   caches; OR rename `clear_caches_on_exit` to `clear_asm_caches_on_exit` to match what it
   actually does.

### Tier 0.5 — Decide on B1-10 documentation/code drift (~10 min)
6. **D1** — Either reclassify B1-10 as fixed in v4.12.0 (preferred — it IS fixed) and
   rename the test file, OR revert the gbd/mhs/subaperture/optimize fixes to genuinely defer.
   Currently inconsistent.

### Tier 1 — Documentation accuracy (~10 min)
7. **D2** — Fix release notes about Newton fast-path: "fast-path shipped; analytic-normal
   stash reverted".
8. **D3** — Fix release notes about NumPy through_focus_scan H-hoist: it's not a hoist, it's
   inherited speedup from the underlying ASM cache.
9. **D4** — Fix release notes about `through_focus_scan_jax` JIT cache: either implement it
   or remove the claim.

### Tier 2 — Convert to known limitations in CHANGELOG, fix in v4.12.1
- **S1** — `io/storage.py` append-side complex64 demotion.
- **S2** — `io/codegen.py` aperture stop drop + 1.31 µm wavelength default.
- **S3** — `ghost.py` docstring R_i/R_j convention conflict.
- **L1** — JAX kernel cache unbounded growth.
- **L2** — JAX path complex64 hard-cast ignores `set_default_complex_dtype`.
- **L3** — `PropagationResult` missing `dy` field.
- **L4** — Sibling-gap pattern instances (mirror guard, GS sibling dispatchers,
  v4.12.0/v4.12.1 label drift).
- The 346 `except Exception:` cleanup (L5).
- `apply_mirror` array-namespace dispatch + `dy` parameter (L6).
- Benchmark hygiene (`through_focus_scan_jax` cold-call cache pollution; v4.12.0 baseline
  JSON not saved; `pyfftw` skip; cross-platform path; README typo; OOM skip).

### Tier 3 — Post-release v4.12.1+ work (documented gaps, not blockers)
- Round-4 coverage gaps (14 fixes still without pinning tests).
- `error_reduction`/`hybrid_input_output` JAX-dispatch `initial_guess` forwarding.
- `gerchberg_saxton(backend='jax', return_history=True)` 2-tuple vs 3-tuple drift.
- Cross-backend NumPy↔JAX phase-retrieval parity test (round-3 carry-over).
- v4.7 kwarg-rename deprecation paths (`lens_prescription` → `prescription`, `_m` suffix
  drop, `wavelength` defaults removed) — `warn_deprecated_kwarg` helper exists but is
  unused.
- The four round-4 untouched findings (apply_real_lens_traced_jax dtype, apply_mirror sign
  R convention, JaxRayState no error_code, save_field_h5 preserve_dtype default).

---

## Audit-process meta-findings

### Test quality trajectory (~10× improvement over the audit series)
| Round | Wrong-reason rate |
|---|---|
| Round 2 | (counted differently — dead-on-arrival fixes were the main concern) |
| Round 3 | 33% (3 of 9) |
| Round 4 | 22% (2 of 9) |
| **Round 5** | **3% (4 of ~125 across 9 new files)** |

The v4.12.0 audit-response is the first in the series where **every claimed round-4 fix has a
pinning test** AND **most tests use real independent ground truth** (Cassegrain hand-derived
S4, C-LR-1 vs `apply_real_lens_traced`, RS-vs-ASM, etc.).

### Cross-audit convergence is high
Three findings were independently caught by 2+ agents in round 5:
- `test_band_limit_uses_strict_less_than_numpy` is weak (Tier-1 + test-quality)
- NumPy through_focus_scan has no real H-hoist (pyFFTW+through_focus + deferred+cross-cutting)
- B1-10 shipped but documented as deferred (WIP + deferred+cross-cutting + PyPI-release)

This is a useful sanity check: when 3 independent agents converge on the same finding from
different angles, the finding is robust.

### The audit-process itself remains useful
Round 1 found ~100 issues. Round 2 found 6 dead-on-arrival fixes + 5 new bugs. Round 3 found
~120 issues. Round 4 found ~75 issues with ~15 release-blockers. Round 5 finds 4 release-
blockers (~30 min total fix time) plus a small list of documentation drift and silent-data-
loss bugs. **Audit yield is decreasing as expected for a maturing codebase**; the remaining
findings are increasingly concentrated in (a) sibling-code gaps, (b) untouched edge-case
modules, and (c) documentation/code drift rather than physics bugs.

---

## Bottom line

**v4.12.0 physics correctness work is genuinely strong.** Round-4 closure is real;
~20 release blockers properly fixed with bit-near-exact pinning tests; Tier-1 perf
optimizations deliver ~10× with no precision regression. The remaining 4 release-blockers
are all ~30-min fixes and target the same brand-damaging first-impression class
(`pip install ... [dev]` + `pytest benchmarks/` fails; `la.set_fft_auto_promote(False)`
raises AttributeError) that round-4 was supposed to close.

**Recommendation: HOLD release for ~30 min, fix A1-A5 (and reconcile D1-D4), then ship.**
Push the remaining items (S1-S3, L1-L8) to a v4.12.1 known-limitations CHANGELOG section.

The Section D items (release-notes/code drift on B1-10, Newton fast-path, through-focus
hoist, through-focus-jax JIT cache) are documentation accuracy fixes that take longer than
A1-A5 but are still hours, not days. Either roll them in pre-tag (preferred) or commit to
the v4.12.1 known-limitations as "release notes incorrectly described X; actual behavior
is Y".

The fundamental engineering is solid. Don't undermine ~70 round-4 closures and a real ~10×
perf delivery with a 30-min papercut.

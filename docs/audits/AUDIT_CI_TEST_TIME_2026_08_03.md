# Audit — CI test-suite time & flakiness (2026-08-03)

**Scope:** `.github/workflows/unit-tests.yml` + `.github/workflows/validate.yml`, and an adversarial,
file-by-file pass over all 445 files in `tests/unit/` (6 chunks of ~74 files, Opus subagents, real
timing where the machine allowed it). **No repository files were modified by this audit** — it is a
findings + recommendations document; every change below is a proposed diff for the author to apply
and verify.

**Method note (read before trusting any number in this doc):** the audit ran on a single 24-core /
34 GB desktop, and for part of it 5–6 heavy pytest-running subagents were active simultaneously,
which **caused an OOM machine restart mid-audit**. Every absolute-second figure below was re-derived
on an *idle* box after that restart (or is explicitly marked as contention-inflated/estimated). Two
independent chunks additionally cross-checked timings at 2–4 pinned BLAS threads specifically
because that is the closest available proxy for a real GitHub Actions runner (2–4 vCPUs); those
cross-checks are what let this report avoid the single biggest trap in an audit like this one —
see §1.

---

## 0. Executive summary

- **Per push/PR, CI currently launches 25 separate job-runs**: `unit-tests.yml` = 12 (`unit`: 4 Python
  versions × 3 pytest-split shards) + 1 (`lint`) + 1 (`mypy`) + 1 (`public-api-smoke`) + 1 (`jax-unit`)
  + 3 (`slow-tests`: 1 Python × 3 shards) = 19; `validate.yml` = 2 OS × 3 Python = 6. Total **25**.
- **`.test_durations` (the file `pytest-split` uses to balance shards) is not usable as a time
  reference.** It totals **316,585 units across 11,223 entries — 87.9 hours** — for a suite that
  runs in tens of minutes. It is not a uniformly-scaled corruption either: two chunks independently
  measured its implied "units per real second" varying by **~26–74× file to file** (§2). Any
  shard-balance decision made from it today is close to noise.
- **The single most important individual finding is a genuine bug, but it is NOT the free win it
  first appeared to be.** 265 files carry a per-file "pin BLAS to 1 thread" guard that is dead code
  (`tests/conftest.py` imports numpy before the guard runs) — the fast `unit` job genuinely runs
  unpinned, multi-threaded BLAS, and the repo's own comments document that this exact condition
  caused nondeterministic eigensolve results before. **However**, a later, more careful chunk proved
  that forcing single-threaded BLAS is not a universal speedup: on large-matrix eigenproblems it
  measured **2.76× SLOWER**, while on many-small-solve files it measured up to ~14–48× faster on
  this desktop (§1). **Do not apply a blanket single-thread pin to the fast gate — this needs
  per-file or per-job-tier treatment and a real CI-hardware A/B test**, not a copy-paste fix.
- **A second, independent sharding defect**, distinct from the corrupted values: `pytest-split`
  balances shards using *call-phase* duration only, so files whose cost is in **collection**
  (repo-wide `ast.parse`/`rglob` walks at import time — 6-11 separate implementations found) are
  invisible to the balancer and can silently overload one shard (§2).
- **~35-45 concrete, itemized test-level recommendations** across the 6 chunks, almost all either
  (a) `@pytest.mark.slow` on a heavy test currently running on the 4×-cost fast gate, (b) sharing a
  byte-identical duplicate computation via a fixture/cache, (c) shrinking a parameter that is
  measured to be far inside its own test's tolerance margin, or (d) removing genuinely dead work.
  **No historical regression assertion is recommended for deletion anywhere in this audit.**
  Itemized in §4, by chunk, with file:line, measured/estimated cost, confidence.
- **Two concrete bug/flake fixes independent of speed**, worth doing regardless of the CI-time
  goal (§3): a test that false-fails whenever `pytest-timeout` is active (its own node ID matches
  its own "no pyfftw thread" assertion), and a jax-import deadlock in `test_audit_misc.py` that the
  workflow currently papers over with a hard-coded `--deselect`.
- **Best-effort total estimate, summed across the 5 chunks that produced a number** (§5): roughly
  **35-50 minutes of CI runner-time per push/PR** from the itemized test-level changes alone
  (excluding the BLAS question, which needs a real A/B and could be a net win or a net loss
  depending on how it's applied — see §1).

---

## 1. The BLAS-pinning finding — a real bug, a documented flake mechanism, and NOT a blanket speedup

### 1.1 The mechanism (confirmed, high confidence)

265 of 444 files in `tests/unit/` open with:

```python
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
```

This is **dead code under pytest**. `tests/conftest.py` does `import numpy as np` / `import lumenairy
as la` at module scope, and pytest imports `conftest.py` **before** any test module — OpenBLAS has
already read its thread count from the environment by the time this guard executes. Verified
directly: `threadpool_info()` reports the full core count both before and after the assignment runs.

`.github/workflows/unit-tests.yml` confirms the asymmetry: `jax-unit` (this job already pins
`OMP_NUM_THREADS=1` etc. in its `env:` block, added to fix a documented JAX/OpenBLAS nested-OpenMP
deadlock) and `slow-tests` (same) both pin. **The `unit` job — 4 Pythons × 3 shards, the dominant
CI cost — does not.** The repo's own comment (lines ~440-450) documents that unpinned BLAS produced
nondeterministic eigensolve results in the past ("byte-identical code passed one run, failed the
next"). **That exact, previously-"fixed" flake mechanism is live in the fast lane on every PR today.**

### 1.2 Why "just copy the env: block" is the wrong fix — measured, not assumed

Three chunks measured this independently, and the story is genuinely two-sided:

| Evidence | Effect |
|---|---|
| `test_niche_audit_w6_bor.py::test_w6_b1_...` (chunk 3) | 302.8 s unpinned → **10.5 s** pinned (~29×) |
| `test_v5_11_0_pmm_anisotropic.py` (chunk 5) | 897 s unpinned → **65 s** pinned (~14×) |
| `test_audit_dynameta_consumer_api.py` (chunk 1) | 340.3 s unpinned → **10.3 s** pinned (~33×) |
| **Chunk 1's control run: same file at `OPENBLAS_NUM_THREADS=2` and `=4`** | **14.0 s and 15.7 s** — i.e. *already close to the fully-pinned time*. **The 33× pathology is a 24-core-desktop artifact; a 2-4 core GitHub Actions runner will not see anywhere near that multiplier.** |
| **`test_v5_6_rcwa_convergence.py` (chunk 6, the heaviest file in its 75-file range)** | **317.2 s pinned vs 115.1 s unpinned — 2.76× SLOWER pinned.** |
| `test_v5_20_6_rcwa_jones_2d_li.py` (chunk 6) | 32.1 s pinned vs 18.7 s unpinned — 1.72× slower pinned |
| `test_v5_6_1_rcwa_symmetry.py`, `test_v5_8_0_pmm.py`, `test_v5_7_0_rcwa_asr.py` (chunk 6) | pinned IS faster here (0.55-0.73× the unpinned time) |
| Most jax-bound / FFT-bound / subprocess-bound files (chunk 6) | ratio ≈ 1.00 — indifferent either way |

**The sign of the effect depends on problem shape, not just on "is BLAS pinned."** Single-threading
helps when the workload is many small (over-subscribed, thread-spawn-dominated) solves, and hurts
when it is one genuinely large eigendecomposition that benefits from multi-threaded BLAS. Chunk 6's
own 75-file range nets out **negative** for a blanket pin (1030 s pinned vs 814 s unpinned, +27%
*worse*), which is the opposite sign of chunks 1/3/5's headline framing.

### 1.3 Recommendation

1. **Fix the dead-code guard's flakiness mechanism regardless of speed** — this is a correctness/
   determinism issue independent of CI-minutes. Either delete the ineffective per-file
   `os.environ.setdefault` blocks (265 of them) and move a single, deliberate pin into
   `tests/conftest.py` **above** its `numpy`/`lumenairy` imports (so it actually takes effect for
   *every* test file, decided once, in one place) — or explicitly decide per-tier as below.
2. **Do NOT blanket-pin the fast `unit` job to 1 thread.** Given §1.2's evidence, run one real
   CI-hardware A/B (push a branch with the `unit` job's `env:` set to `OMP_NUM_THREADS=1` etc.,
   compare wall time and any behavior change against 5-10 baseline runs) before deciding. A likely
   better middle ground, worth testing first: pin to a **small fixed thread count (2-4)**, matching
   actual runner core count, rather than 1 — chunk 1's control data suggests this captures most of
   the flakiness fix without the large-matrix penalty chunk 6 found at strict single-threading.
3. Whatever is decided, apply it **consistently** — chunk 5 found ~25 of its 74 files carry the
   dead guard and ~10 heaviest files (including the ones that most need it) don't carry it at all,
   so even the guard's *intent* is inconsistently expressed today.

---

## 2. `.test_durations` / sharding — two independent defects, not one

1. **Collection-time blind spot (chunk 4).** 6-11 files across the suite independently
   `rglob('*.py')` + `read_text` + `ast.parse` the ~220-file `lumenairy/` tree **at module-import
   (collection) time**, each with its own private, unshareable `@lru_cache`. Measured: 9 such files
   register only **33.5** recorded call-phase duration units (near-invisible to the balancer) yet
   cost **+47.8 s of real collection time** in one measured comparison. `pytest-split` balances on
   call-phase durations only, so it can and does pile these files into one shard where their true
   cost is invisible until the shard runs. **Fix:** one shared `tests/unit/_source_index.py` module
   exposing a single process-cached `parse_tree(path)` / `lumenairy_py_files()`, replacing the 6-11
   private implementations.
2. **Internally inconsistent scale (chunk 6, independent confirmation).** Measured "units per real
   second" implied by `.test_durations` ranges from **~6.5/s** (`test_v5_3_2_stamp_changelog.py`,
   268 units / 41.1 s) to **~480/s** (`test_v5_7_0_rcwa_asr.py`, 771 units / 2.7 s) — a **~74×**
   range. Concretely, the file believes `test_v5_6_1_rcwa_symmetry.py` (4252 units) is 5.5× heavier
   than `test_v5_21_gbd_maslov_perf.py` (1136 units); measured, it is **7.5× lighter** (8.6 s vs
   64.7 s). The workflow's own comment claims the resulting split is "even to <0.1%" — it is evenly
   splitting **inconsistent weights**, which is worse than not balancing at all in some cases.

**Recommendation:** regenerate `.test_durations` from a clean, idle, single-threaded, single-machine
run (`pytest --store-durations` or equivalent) before trusting `pytest-split` again, and land the
shared source-index fix (item 1) first so the *next* regeneration isn't corrupted the same way.

---

## 3. Two flakiness/correctness bugs found (not primarily about speed)

- **A test that structurally cannot pass once `pytest-timeout` is active.**
  `test_audit_g06_perf.py::test_s5_8f_pyfftw_interfaces_cache_daemon_disabled` asserts
  `not any("pyfftw" in n for n in thread_names)`. `pytest-timeout`'s watchdog thread is named after
  the failing node ID — which contains the substring `pyfftw` because **the test's own name does**.
  Reproduced directly (chunk 1): `AssertionError: a pyFFTW cache daemon thread is running:
  ['mainthread', 'pytest_timeout tests/unit/test_audit_g06_perf.py::test_s5_8f_pyfftw_...']`. This
  currently blocks ever safely adopting `--timeout` on the main gate as a hang-guard. Fix: filter
  the `pytest_timeout` /`pytest-timeout` watchdog thread name out of the check, or match on the
  actual pyFFTW daemon thread's real name prefix instead of a substring search.
- **A jax-import deadlock in `test_audit_misc.py`.** Hits the same JAX/OpenMP/OpenBLAS nested
  deadlock the `jax-unit` job's pinning was added for, but from a file that is not itself
  jax-selected — the workflow currently works around a specific symptom of this with a hard-coded
  `--deselect` of one cookbook test. Recommend splitting the jax-guarded tests in that file into
  their own module so the deselect line can be removed instead of accumulated.

---

## 4. Itemized test-level recommendations, by chunk

Legend: **[M]** = measured this audit, **[E]** = estimated from reading (not executed), confidence
as stated by the auditing agent. All estimated savings are **per relevant job-tier**, not yet
multiplied across the Python/shard matrix unless stated.

### Chunk 1 — `test_analysis.py`, `test_analytic_ray_transfer.py`, `test_ao_dm.py` + 71 `test_audit_*.py`

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| Cheap question, dense O(work) test | `test_audit_g04_guards_prop.py::test_s2_20_reconstruct_dense_footgun_warns` | 47.2 s **[M]** | monkeypatch `_GBD_DENSE_WARN_WORK` down; use a tiny bundle instead of `n=30000` beamlets on 64×64 | ~47 s × fast-gate multiplicity | high |
| Over-provisioned solve size | `test_audit_v5_24_2_b2_bor_exports.py::test_borstack_solve_return_shape_unchanged` | N=64 (0.75 s warm/9.95 s cold) **[M]** | N=32 — identical 6-mode set, *better* closure (1.55e-15 vs 7.33e-15) | most of the test | high |
| Duplicate heavy solves | `test_audit_dynameta_consumer_api.py` — two tests build identical `_pmm_grating`+`_rcwa_grating(30°,25°)` | — | module-scoped fixture cache | 1 solve | high |
| Duplicate solves | `test_audit_p1_bor_flux.py` — `_solve(1.0)` / `_solve(1e-6)` each computed 2× across 3 tests | — | `lru_cache` | 2 of 6 `BORStack(N=80)` solves | high |
| Over-provisioned margin | `test_audit_glass.py::test_seidel_correction_field_matches_traced_within_few_waves` | 27.0 s of 32 s **[M]** | N=96→48, dx=120e-6 (keep N·dx ≥ aperture radius); ~200× discriminating margin measured | ~4× cheaper | medium (needs 1 confirming run) |
| Misclassified as fast, unmeasured tail | `test_audit_lens_models_2026_07.py` | 381 s in contended sweep; clean re-run **timed out at 300 s** inside `test_a1_auto_n_v2_resolves_demanding_default_quadrature` | biggest fast-gate item in the chunk with **zero** `slow` marks; needs full breakdown | unknown, likely large | flagged, not resolved |

### Chunk 2 — audit/eme/fga/hammer group

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| Misclassified as fast | `test_fga.py` (whole file) | ~7 min, author's own figure **[E]** | `pytestmark = pytest.mark.slow`; the file's own in-code justification cites `xdist --dist loadfile`, which the workflow no longer uses (grep repo-wide for the same stale justification) | ~7 min × 3 of 4 Pythons | high |
| Misclassified as fast | `test_g2_displaced_congruence.py::test_congruence_beats_collimated_on_doublet` (N=4096, unmarked; sibling at same N *is* slow) | did not complete under contention | mark `slow` | — | medium |
| Misclassified as fast | `test_hammer_h1_slant_obliquity.py::test_h1_slant_lands_in_oracle_window_at_converged_sampling` (N=4096, only unmarked test in the file) | — | mark `slow` | — | high |
| Misclassified as fast | `test_hammer_h7_gbd_diverging.py` — 3-param GBD at N=768/bpa=128 × 15-pt focus scan | — | mark `slow` on the 2 heaviest | — | medium |
| Byte-identical duplicate eigensolves | `test_eme_2d_vector.py` (already `slow`) — `layer_vector_modes(...)` identical args at 3-4 call sites; `_oracle_band(...)` identical at 2 sites | ~2 min file **[E]** | module fixtures + `lru_cache` on `_oracle_band` | ~45 s | high |
| Byte-identical duplicate | `test_eme_2d.py` (already `slow`) — `layer_modes(...)` identical at 3 sites | ~2 min file **[E]** | same pattern | ~25-30 s | high |
| Byte-identical duplicate | `test_coupled_eigensolver.py` (already `slow`) — `guided_modes(...)` / `radial_coupled_modes(...)` each duplicated; a 4th PML solve redundant with 3 already-solved σ values | — | dedup 3 of 9 dense eigensolves | ~33% of file | high |
| Byte-identical duplicate | `test_bor_solve.py` (already `slow`) — `build_layer(...)` rebuilt identically twice; one test pays a full eigensolve for a 1-line property of an already-computed spectrum | — | dedup 1 of 7 eigensolves | — | high |
| Byte-identical duplicate | `test_gate4.py::_gate4_compare(Lam=3.0, Rfac=16, pol='tm')` — identical args in 2 tests | file's own figure ~25-45 s **[E]** | `lru_cache` | ~1/3 of file | high |
| Byte-identical duplicate | `test_carrier_referenced.py` — both tests build an identical N=4096 ground truth, both **unmarked** (×4 Pythons) | — | module fixture | ~5-15 s × 4 | high |
| O(N) algorithmic waste — appears in ≥3 files | whole-grid encircled-energy loop `[I[r<=rr].sum() for rr in rb]`: `test_g2_displaced_congruence.py:172`, `test_hammer_h2_displaced_projection.py:121`, `test_fga_h4_h5.py:71` | **11.08 s → 0.059 s measured (189×)**, agrees to 1.2e-7 | crop-then-sorted-cumsum rewrite; also cuts several GB peak RSS at N=8192 | ~90 s combined **[M]** | high |
| Cheap-question/expensive-test | `test_niche_audit_w3_propagators.py::...test_fallback_survives_mismatched_output_shape` | 23.2 s (10.82+6.57+5.85) **[M]** | drop the `(64,64)` case — already covered more strongly by a sibling test at both 32/64 | 10.8 s × 4 Pythons | high |
| Over-provisioned negative-result test | `test_eme_diffraction.py::test_structured_layer_does_not_converge` | 4 solves for extrema-only assertions | keep M=(1,4) or (1,2,4) | 25-50% of dominant test | medium-high |

### Chunk 3 — `test_niche_audit_w6-w9_*`, `test_niche_{c,d,e4,k,p,r,s}*`, plus the perf/optimize misc files

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| Misclassified as fast | `test_niche_audit_w6_bor.py::test_w6_b1_default_path_is_bit_identical_to_explicit_rbig` | **130.88 s of 138.6 s (94%) [M]** | mark `slow`; separately N=60→17 (measured 0.05 s vs 0.19 s, N-independent claim) | ~131 s × 3 of 4 Pythons | high |
| Misclassified as fast, whole file | `test_niche_audit_w6_eme.py` (zero `slow` marks — the exact class the slow gate exists for) | **197 s pinned [M]** | mark the 4-5 eig-heavy tests `slow` | most of 197 s | high |
| Misclassified as fast | `test_niche_e4_corrected_relay_oracle.py` | ~110-130 s **[E]**, file's own figure ~8 s/call × 14-15 cells | mark 3 cliff-guard tests `slow` | ~110-130 s | medium |
| Misclassified as fast | `test_niche_p2_design_battery.py` | ~20 N=1024 chain runs **[E]** | mark `slow`, or keep 4 representative cells fast | most of file | medium |
| Misclassified as fast | `test_niche_d5_dx_flatness_gate.py` | ~10-14 chain runs **[E]** | mark the 5-case `test_gate_has_teeth` `slow`; keep the single tripwire fast | most of file | medium |
| Flaky + expensive wall-clock assertion | `test_niche_k3_perf.py::test_remap_2d_shared_delaunay_is_faster` | 5 reps × 2 Delaunay paths, N=768 **[E]** | move to `bench` marker (already defined) or `slow`; correctness already covered by a sibling byte-identity test | full test | high |
| Over-provisioned parametrize | `test_niche_audit_w8_shapes.py::test_w8_efficiency_convergence_to_the_analytic_answer` | 24 cases | a sibling test already treats 2 combos as adequate for the same claim → 6 cases | ~75% of group **[E]** | medium |
| Near-duplicate 3× pinned contract | `e4::test_e4_cliff_guard_recovers_oversized_aperture`, `p2_design_battery::test_battery_cliff_cells_need_the_guard`, `p2_design_battery::...is_a_focal_catastrophe` | 3 independent heavy runs of the same claim | keep 2, retire or fixture-share the 3rd | ~1 heavy run | medium |
| Duplicated oracle infrastructure | `test_niche_r7_intragroup_curvature.py` ↔ `test_niche_r8_tiltaware_chain_api.py` — byte-identical `_group_geometry`/`_oracle_phase`/`_exit_rms`/`_TRIPLET`, same parametrize pair | — | shared helper + session-scoped default-path cache | ~2 N=1024 runs | medium |
| Cheap fix, zero coverage loss | `test_perf_v4_12_0_asymptotic.py` 3-test cluster with redundant O(N⁴) scalar references | — | merge into one parametrized test | small | low priority |

### Chunk 4 — dispatcher-pin / walker / `agent_[a-g]` files + `test_rcwa.py`/`test_propagation.py`/`test_raytrace.py`/etc.

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| The single biggest item in the chunk | `test_rcwa.py::test_analytic_energy_and_clean_convergence` | ~34-44% of the whole 74-file chunk by `.test_durations` ratio; **blew a 300 s pytest-timeout** in one run; M sweep (6,10,14) → eig dim 338/882/**1682**, O(n³) means M=14 alone ≈ 87% of the sweep | (6,8,10) — ~6× cheaper, both assertions (exact energy conservation + monotone convergence) survive; the file's own docstring says it should run "quickly (small truncation)" | ~6× of the file's dominant test **[E, needs 1 confirming run]** | medium-high |
| Related, cheaper | `test_rcwa.py::test_analytic_metal_disk_positive_absorptance` | M=8 (578² eig) to assert only `A > 1e-3` | M=4 | ~45× less eig work on this one test | medium-high |
| Nested-subprocess pattern (very expensive per assertion) | `test_v4_15_4_agent_b.py` — 2 of 4 subprocess tests are strict subsets of a 3rd | 197.6 of 197.8 s file total | delete the 2 subsumed tests | ~53% of file | high |
| Nested-subprocess pattern | `test_v4_15_2_agent_a.py::...test_changelog_test_count_arithmetic_reconciles` — spawns **10** separate `pytest --collect-only` subprocesses | 60.4 of 61.2 s file total | one subprocess collecting all 10 files at once | ~90% of that test | high |
| Dead code | `test_v4_15_4_agent_c.py::_pytest_collect_only_count` | runs `pytest --collect-only` over all of `tests/unit`, has **no caller** since the v4.16.1 rewrite | delete | full cost of an unused helper | high |
| Duplicate physics per parametrize case | `test_s5_5_jones_field_bridge.py` — `_rcwa_grating(12°)` solved 4× identically across 4 tests | 305 of 379 s file total | module-scoped fixture / `lru_cache` keyed on theta | ~60% of file | high |
| Misclassified as fast, deliberate slow reference | `test_v4_15_agent_a.py::test_modal_asymptotic_perf_win` | 258.5 of 323 s file total — deliberately runs the slow warm-start reference to prove a 5× speedup | mark `slow`/`bench`; 7 sibling tests already cover correctness | ~80% of file | high |
| Over-provisioned parametrize | `test_slant_chunk_byte_identical.py::test_slant_stop_clearap_band_byte_identical` | 175.5 s / 24 items — N=1024 dimension exists only to hit a gate a sibling test already covers at both N | drop N=1024 here → 12 items | ~80% of that test | high |
| Redundant parametrize points | `test_slant_chunk_byte_identical.py::test_multiple_band_sizes_agree` | 305 s, 9 `chunk_rows` values, several mutually redundant | drop `3` (redundant w/ `1`+`70`) and `128` (redundant w/ `64`) | meaningful fraction of 305 s | medium |
| Systemic — collection-time blind spot | 6-11 files independently walk/parse the `lumenairy/` tree at import time (also **§2** finding) | +47.8 s measured collection time, ~invisible to `pytest-split` | one shared cached `tests/unit/_source_index.py` | 40-70% of that collection cost | high |

### Chunk 5 — RCWA/PMM `test_v5_10-v5_20_*` heavy files

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| Biggest single item in the whole audit | `test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_crossed_cell_converges_and_beats_laurent` | **~615 s [M via scaling]** — cost is `(2M+1)^6`; the `ref = sumR(17, "fff_nv")` line alone ≈ 475 s | drop the reference from M=17 → M=13 (2 ladder steps of margin remain; assertions are relative/monotone, not absolute) | **~375 s** | medium-high, needs 1 confirming run |
| Over-provisioned, zero accuracy return | `test_v5_11_0_pmm_segments.py::test_segments_out_of_plane_matches_rcwa` (53.4 s) + `pmm_anisotropic.py::test_pmm_1d_out_of_plane_and_errors` (10.3 s) | `elements_per_region=6` **measured** to give the *same* per-order error (2.39e-4 vs bar 1.5e-3) as `epr=3` — error is oracle-truncation-limited, not element-limited | `elements_per_region` 6→3 | **~30 s [M]** | high |
| Over-converged by the test's own docstring | `test_v5_12_0_pmm_slant_and_convergence.py::_oop_staircase(n_slabs=200)` | docstring states n_slabs is converged at 7e-6 against a 3.8e-3 bar (500× margin) | n_slabs 200→64 (still 50× under bar) | ~37 s **[E, 1 confirming run]** | medium |
| Order count driven by an overflow guard that can be met more cheaply | `test_v5_11_0_rcwa_internal_field.py::_deep_metal_stack(n_orders=120)` | guards `exp(+lam·k0·z)` overflow at `Re(lam)·k0·thick > 709`; current thick needs m>71 hence n_orders=120 | `thick` 0.6µm→1.8µm drops the threshold to m>23; n_orders 120→30 still overflows with margin; eig 482²→122² (~60×) | ~26 s **[E, 1 confirming run]** | medium |
| jax leg is compile-bound, not solve-bound (non-obvious, changes what "reduce parametrize" means) | `pmm_jones_autodiff`, `pmm_autodiff`, `rcwa_2d_autodiff` — first case in a new cell/geometry/FOM costs 10-20s to *compile*; later cases in the same program cost ~2s | see file for 6 itemized sub-changes (fewer distinct CELLS/geometries, not fewer parametrize rows; one `jax.grad` call already returns all 4 gradient entries the test currently re-derives 4×) | **~120-135 s of ~475 s measured jax-leg total (25-30%)** | medium-high |
| Dead work | `pmm_anisotropic.py::test_no_accuracy_floor_jones_error_improves` | computes degrees 8-24 (17 solves), only reads d≤12 and d≥16 — degrees 13-15 solved and discarded | drop the 3 unread degrees | 2.7 s **[M]**, small | high |
| Misplaced gate | `pmm_autodiff.py::test_numpy_path_unchanged_normal_and_oblique` | 16 pure-numpy solves sitting inside a module-level `importorskip('jax')` file → never runs on the 4×3 fast matrix, only on the single jax runner where it tests nothing jax-related | move out of the jax-guarded file | recovers real fast-gate coverage; time-neutral | high |
| Unmeasured, flagged as highest remaining risk | `test_v5_11_0_rcwa_fff_nv_2d.py::test_fff_nv_metal_stripe_matches_1d` | y-uniform stripe solved at `n_orders_y=16` (1089 harmonics); a sibling in `rcwa_audit_fixes.py` already proves `n_orders_y=2` is valid for the same physics | `n_orders_y` 16→2 | potentially **minutes**, unverified | flagged only |

### Chunk 6 — `test_v5_20_13...test_zcascade.py` (75 files, solo re-run after the restart)

| Finding | File:line | Cost | Change | Est. saving | Confidence |
|---|---|---|---|---|---|
| Marker inversion — the file's *only* `slow` mark is on the wrong test | `test_v5_21_maslov_jax_caustic.py` | 3 unmarked tests = **174.6 s of 190 s [M]**; the file's only jax tests (the ones its name implies) are the cheap 6.9+3.5 s | mark the 3 heavy NumPy tests `slow` | **~175 s × 3 of 4 Pythons ≈ 8.7 min** | high |
| Dead-work + logic fix (net removal, not a lane move) | `test_v5_3_2_stamp_changelog.py::test_dry_run_against_current_changelog` | runs the same 20 s subprocess **twice**; the "before" snapshot is captured *after* the first subprocess, so it never actually tests what it claims | move the snapshot before the first (only) subprocess call — strictly better coverage, one subprocess | **~20 s × 4 Pythons = 80 s**, true deletion | high |
| Misclassified as fast | `test_v5_21_gbd_maslov_perf.py` — 2 dense-reconstruction tests | 49.8 of 64.7 s **[M]** — N=128 dense O(N⁴) sum, exact identity to <1e-9 vs the FFT path | mark `slow` | ~2.5 min | high |
| No `slow` mark anywhere despite being the #2 heaviest file in `.test_durations` | `test_v5_6_rcwa_convergence.py` | 309 of 317 s pinned / ~112 of 115 s unpinned **[M]** across 3 convergence-study tests | mark all 3 `slow` | ~5.6 min | high |
| Marker on the wrong test (again) | `test_v5_21_pmm2d_staggered_oblique.py` | the marked test costs 0 s (deselected); 2 unmarked tests cost **53.8 of 56.1 s [M]** | move the `slow` mark to the 2 actually-heavy tests | ~2.7 min | high |
| Misclassified as fast | `test_v5_21_gbd_windowed_adaptive.py::test_soft_edge_improves_hard_aperture_focus` | 27.6 of 40.7 s **[M]** | mark `slow` | ~1.4 min | high |
| Misclassified as fast | `test_v5_21_lens_accuracy_extensions.py::test_traced_multibranch_matches_exact_diffraction_oracle` | 25.1 s **[M]**, pure NumPy, 2.56M-ray launch grid | mark `slow` | ~1.25 min | high |
| Misclassified as fast, weak assertion | `test_v5_2_3_subaperture_image_plane.py` — 2 magnification tests | 16.6 of 26.5 s **[M]**; one test's own docstring calls its bar "a weak floor" | mark `slow` | ~50 s | high |
| **Coverage hole, not a saving** — jax tests silently never run anywhere | `test_v5_21_maslov_jax_caustic.py`, `test_v5_21_lens_accuracy_extensions.py`, `test_v5_21_gbd_maslov_perf.py` | these guard with `@pytest.mark.skipif(not _jax_ok(), ...)` instead of `importorskip`, so the `jax-unit` job's file-selection grep (`importorskip.{0,4}jax`) never selects them — their jax tests silently skip in the no-jax fast gate and never run anywhere | fix the guard idiom to be selectable, or add these files to the job's selection explicitly | adds ~50 s to the cheap unsharded jax lane; **closes a real coverage gap** | high |

---

## 5. Total estimated impact

| Gate | Estimated saving | Basis |
|---|---|---|
| Fast `unit` (4 Pythons × 3 shards, dominant cost) | Chunk 2 ≈ 2.5-3.5 min/Python · Chunk 3 ≈ 2-3 min/Python · Chunk 4 ≈ several min (test_rcwa.py alone potentially the largest single item, unconfirmed) · Chunk 6 ≈ 22.4 min total (measured) | mostly `@pytest.mark.slow` moves — near-zero coverage risk |
| Slow-tests (1 Python × 3 shards) | Chunk 2 ≈ 4-6 min · Chunk 5 ≈ 7 min (R1 alone) · Chunk 6 absorbs +144 s from the moved tests | net still large, gate has more headroom (30-35 min cap vs currently-tighter fast gate) |
| jax-unit (1 Python, unsharded) | Chunk 5 ≈ 25-30% of ~475 s measured jax-leg time (~2 min) · Chunk 6 fixes a coverage hole for +~50 s | |
| **Best-effort combined total** | **roughly 35-50 minutes of CI runner-time per push/PR**, concentrated in `@pytest.mark.slow` moves with near-zero coverage risk, plus the two true deletions (R2/chunk 6, and the nested-subprocess fixes/chunk 4) | excludes the BLAS question (§1), which needs a real A/B and could be a net win OR a net loss depending on how it is scoped |

**This is a floor, not a ceiling.** Three files across the 6 chunks were flagged as "likely large,
unmeasured" and deliberately not counted here: `test_audit_lens_models_2026_07.py` (chunk 1),
`test_niche_audit_w7_pmm.py`/`w7_rcwa.py` (chunk 3, unverified "cheap per solve" assumption),
`test_v5_11_0_rcwa_fff_nv_2d.py::test_fff_nv_metal_stripe_matches_1d` (chunk 5).

---

## 6. Suggested order of operations

1. **Land the two zero-risk correctness fixes first** (§3) — they're small, independent of
   everything else, and directly address "CI is flaky."
2. **Land the `@pytest.mark.slow` moves** (the large majority of §4's items) — each is a 1-line
   change, near-zero coverage risk, and collectively the biggest lever that doesn't require any
   physics judgment call.
3. **Land the byte-identical duplicate-solve fixture/cache fixes** — mechanical, verifiable by
   diffing output before/after, no tolerance judgment needed.
4. **Do the ~8 "needs 1 confirming run" parameter shrinks one at a time**, each verified against its
   own test's actual measured tolerance margin (do not batch these — a few are genuinely
   borderline and deserve individual attention, e.g. chunk 5's R1 dropping M=17→13).
5. **Regenerate `.test_durations` from a clean run** (§2) only after step 2 has changed which tests
   are in which gate, and only after the shared source-index fix (chunk 4) is in, so collection time
   becomes visible to the balancer for the next regeneration.
6. **Run the BLAS A/B last, deliberately, on real CI hardware** (§1.3) — it is the single biggest
   possible lever in either direction and the one place in this whole audit where desktop timings
   are not a reliable guide to the production answer.
7. **Follow up, not blocking:** the 3 unmeasured "likely large" files noted in §5, and chunk 3's
   flagged-but-unverified `w7_pmm`/`w7_rcwa` assumption.

---

## 7. Methodology / integrity notes

- All 6 chunk audits explicitly avoided recommending deletion of any historical regression pin or
  assertion; every item is a marker move, a fixture/cache share, a measured-safe parameter shrink,
  or removal of demonstrably dead work.
- Two chunks' first attempts were invalidated by real infrastructure hazards worth remembering for
  next time: (a) `TaskStop` on a backgrounded pytest process **does not kill the underlying pytest
  process** on this platform — chunk 6 found an orphaned process still holding ~5,575 CPU-seconds
  hours after being "stopped," contaminating an earlier timing pass; (b) `pytest-timeout`'s default
  thread-based kill method calls `os._exit()` on Windows, killing the **entire** run rather than
  just the offending test — do not use `--timeout` for exploratory timing runs on this platform.
- Absolute seconds throughout this report were measured on Windows / Python 3.14 / a 24-core desktop
  with jax installed; production CI runs ubuntu / Python 3.10-3.13 / 2-4 vCPUs without jax on the
  fast gate. Relative rankings and marker-inversion findings (a test with the *only* `slow` mark on
  the *cheap* test in its file, etc.) are platform-independent; absolute-second and BLAS-ratio
  figures are not, and are labeled as such throughout.

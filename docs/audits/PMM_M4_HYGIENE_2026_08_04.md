# M4 -- Hygiene: two pre-existing local failures + the eight staleness corrections

**Date:** 2026-08-04 - **Branch:** `feat/pmm-per-layer-roadmap` (M1 committed at `d30f1ca`)
**Plan:** `docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md`, mission M4
**Scope delivered:** the two user-ordered test failures (F-1, F-2) + D-1 (the eight
documentation corrections S-1..S-8). T3-7 and T3-5 are NOT delivered -- see S4.
**Concurrency:** `lumenairy/elements/pmm/**` is owned by the concurrent M2 mission and was
NOT edited by this mission. Everything below in `lumenairy/` is `elements/rcwa/`.
**Builds:** Windows (py3.14.6, numpy 2.4.4, scipy-openblas 0.3.31, 24 threads) and
WSL Ubuntu (py3.12.3, numpy 2.4.6, OpenBLAS, no `threadpoolctl`).

---

## 0. Headline: the two failures are ONE mechanism, and they were masking each other

Both assigned failures reduce to a single fact, measured directly:

> **`threadpoolctl`'s BLAS cap is PROCESS-GLOBAL on OpenBLAS, not thread-local.**
> A worker thread entering `threadpool_limits(1, "blas")` takes the *main* thread's reported
> pool to 1 as well, and its exit restores the previous value **for everyone**.

```
main-before                [24]
  inside-worker            [1]
main-while-worker-pinned   [1]     <- the main thread's pool changed
  worker-after-exit        [24]
main-after                 [24]
```

Two consequences, and each is one of the assigned failures.

**(a) N concurrent enter/exit pairs RACE** -- the first worker to finish restores the
environment pool while its siblings are still inside `solve`, so their BLAS reduction order
changes mid-solve. That is **F-2** (byte-identity).

**(b) The unbalanced restores LEAK.** Measured directly, four staggered per-worker caps on a
24-thread pool:

```
start                                  [24]
after unbalanced per-worker caps ->    [1]      <- leaked, permanently
```

The pre-fix sweep therefore left the **whole process** at 1 BLAS thread. And **F-1**'s parity
metric is a function of the BLAS thread count (S1.1). So running the sweep file first silently
pinned the process to 1 thread and the PMM-JAX parity test passed; run alone, the process was
at 24 and it failed. That is the entire "fails alone, passes with its neighbour" signature, and
it is why neither failure could be diagnosed from the failing file alone.

Both are FIXED, by different classes of fix, because the root causes are different: F-2 is bad
code (a real shared-state race, with a documented contract that was false); F-1 is a bad test
(an absolute bar on a magnitude the BLAS thread count sets).

---

## 1. F-1 -- `test_v5_20_2_pmm_jones_2d_jax.py::test_pmm_jones_2d_jax_forward_matches_numpy[inplane]`

### 1.1 Root cause: BAD TEST -- an absolute bar on a BLAS-thread-count-dependent magnitude

The assertion that failed was `abs(sum(T_jax) - sum(T_numpy)) < _PAR_TOTAL` with
`_PAR_TOTAL = 5e-3`. Holding the **code, the build, the geometry and the degree fixed** and
varying **only** `OPENBLAS_NUM_THREADS`, degree 11, Windows/py3.14/numpy 2.4.4 **[M]**:

| channel | 1 thread | 2 threads | 24 threads | bar |
|---|---|---|---|---|
| per-order R | 4.80e-11 | 7.25e-08 | 2.52e-07 | 2e-2 |
| per-order T | 7.11e-06 | 8.04e-03 | 5.41e-03 | 2e-2 |
| total R | 5.95e-11 | 9.53e-08 | 2.94e-07 | 5e-3 |
| **total T** | **1.01e-05** | **3.19e-03** | **1.83e-02** | **5e-3 <- FAILS at 24** |
| Jones (full) | 2.47e-10 | 2.05e-07 | 9.87e-07 | 2.83e-1 |
| Jones sing. values | 9.04e-11 | 1.55e-07 | 3.20e-07 | 5e-3 |

Deterministic within each thread count (3 identical runs at 24). So the bar passes on a 2-core
CI runner and fails on a 24-thread box **with identical code** -- exactly the reported
"passes CI, fails locally". Re-tuning the constant could never fix that, and the history shows
it was tried twice (v5.25.0 `1e-3`, v5.30 `5e-3`).

### 1.2 Which engine moves, by an independent instrument

Lossless cell, two incident polarizations, so `sum(R) + sum(T) -> 2.0` up to truncation. **[M]**

| degree | closure NumPy @1 | closure NumPy @24 | closure JAX @1 | closure JAX @24 |
|---|---|---|---|---|
| 7 | 1.9970982 | 2.0055203 | 2.0123702167183377 | 2.0123702167183732 |
| 9 | 2.0123865 | 2.0123864 | 2.0123864477691695 | 2.0123864477691913 |
| 11 | 2.0125077 | **2.0307687** | 2.0124975650960613 | 2.0124975650961530 |
| 13 | 2.0105244 | 2.0115177 | 2.0125618003701216 | 2.0125618003700976 |

* The **JAX twin is invariant** to the thread count (<= 9.2e-14 at every degree) and smooth in
  degree (2.01237 -> 2.01256). XLA does not use NumPy's BLAS.
* The **NumPy side moves**, and at deg 11 / 24 threads it manufactures ~1.8% extra energy on a
  LOSSLESS cell (2.0125 -> 2.0308). The twin-vs-NumPy total-T gap *is* that manufactured energy:
  R agrees to 2.9e-7, so `gap_T = |closure_np - closure_jx|` identically.

The in-plane arm is the only one affected: the NumPy path takes the symmetric `eig(P Q)` route
for an in-plane cell while the twin keeps the generator route, and that route has the
near-degenerate mode pair. The **out-of-plane arm** (both engines on the generator route) is
<= 3.6e-14 at every thread count -- the null control.

**Degree cannot rescue it.** deg 9 is thread-stable on THIS build (closure moves 1.6e-8) and
v5.30 recorded deg 9 as the *bad* one and deg 11 as clean on CI's build. Picking a degree is
playing the roulette, not fixing it -- which is why the v5.30 fix rotted in one release.

### 1.3 Fix

`tests/unit/test_v5_20_2_pmm_jones_2d_jax.py`. The test now runs **two legs**:

* **Leg 1 (always) -- the build-invariant channels at their historical bars.** Every channel
  except total-T is invariant across 1/2/24 threads with >= 3 orders of headroom (worst:
  per-order T 8.0e-3 against 2e-2). Per-order R/T, the full Jones matrix and its singular
  values keep their v5.25.0 values, asserted unconditionally. The **R half of the era-pinned
  `_PAR_TOTAL` is RETAINED at 5e-3** (measured 2.9e-7).
* **Leg 2 (when the pool can be pinned) -- algorithm agreement, which is what the test's name
  promises.** Both solves run inside `threadpool_limits(1, "blas")`, verified to reproduce the
  `OPENBLAS_NUM_THREADS=1` numbers to ~1e-15 in-process. Every channel is then pinned at
  **1e-3**, i.e. **2-3 orders TIGHTER than any bar this test has ever carried** (measured
  in-plane maxima: 4.8e-11 / 7.1e-6 / 5.9e-11 / 1.0e-5 / 2.5e-10 / 9.0e-11). Leg 2 also asserts
  the twin's own energy closure does not depend on the thread count (`< 1e-9`) -- the
  instrument that identified NumPy as the moving side.
* `threadpoolctl` is an OPTIONAL dependency (it is absent on the WSL build), so `_blas_pin`
  returns `None` there and leg 2 is skipped; leg 1 still runs.

**The era-pin is verbatim** in the test source, with the reason it moved and where the two
halves went. Net: the test is strictly MORE powerful than before -- it lost one bar that could
not distinguish an algorithm defect from the BLAS pool, and gained six that are 2-3 orders
tighter.

### 1.4 Evidence

* Cross-thread table and closure table above, both `[M]` on Windows.
* Cross-build: `tests/unit/test_v5_20_2_pmm_jones_2d_jax.py` -- **14 passed** on Windows
  (24 threads, leg 2 ACTIVE). On WSL/OpenBLAS leg 2 is skipped by construction (no
  `threadpoolctl` -> `_blas_pin` returns `None`), so that build exercises leg 1 only; see S4a for
  the run status. Leg 1's bars are the ones that were already green on every build.
* Null control: the `[oop]` parametrization is <= 3.6e-14 on both legs and both thread counts,
  so the change cannot have loosened anything that was tight.
* Ruff clean (WSL, `ruff check lumenairy/ tests/unit/`).

### 1.5 The CODE defect this exposed -- referred, not swallowed

Independently of the test, the measurement establishes a real numerical defect **[M]**:

> `pmm_jones_2d`'s NumPy in-plane path draws a BLAS-thread-count-dependent answer, and at
> degree 11 on a 24-thread pool it violates lossless energy closure by 3.1% against the same
> solve's 1.25% truncation floor -- ~1.8% manufactured energy, invisible to every per-order and
> Jones bar in the suite.

That file (`lumenairy/elements/pmm/twod_jones.py`) is owned by the concurrent M2 mission, so
this mission did not edit it. It is the **same class** M1 just hardened (a near-degenerate
eigenproblem whose null-space draw is build-dependent) and it belongs in M1/N-2's census, which
covered the per-layer mortar/star/lstsq sites but not this one. **Recommended owner: M1's
conditioning census, extended to `pmm/twod_jones.py`'s in-plane `eig(P Q)` route.** The
reproduction is one command and is recorded in S5.

---

## 2. F-2 -- `test_v5_20_8_rcwa_threaded_sweep.py::test_threaded_sweep_is_byte_identical_to_serial`

### 2.1 Root cause: BAD CODE -- the sweep applied a process-global setting once per worker

`RCWAStack.solve_vs_wavelength` entered `_blas_threads_quiet(blas_per_worker)` **inside**
`_solve_one`, i.e. inside each worker, on the belief -- stated in three places in the source --
that the cap was thread-local:

* `rcwa/stack.py` (in `_solve_one`): `# the BLAS cap is THREAD-LOCAL, so each worker sets its own.`
* `rcwa/_core.py` (above `_BLAS_STATE`): `THREAD-LOCAL: concurrent solves with different caps
  must not leak each other's setting (the context manager's save/restore would otherwise race
  on a shared global).`
* `set_blas_threads.__doc__`: `The setting is thread-local, so concurrent solves with different
  caps don't interfere.`

What is thread-local is only the **request** (`_BLAS_STATE = threading.local()`). **Applying**
it goes through `threadpoolctl`, which on OpenBLAS calls `openblas_set_num_threads()` --
process-global (demonstration in S0). So N workers' enter/exit pairs race on one global, and
the documented contract ("BYTE-IDENTICAL to a serial sweep regardless of worker count") was
false.

The divergence magnitude confirms the mechanism rather than a logic error -- **[M]**, four
consecutive runs of the test's own 12-point sweep, 4 workers vs serial:

| run | max abs dR | max abs dT | max abs dJones |
|---|---|---|---|
| 1 | 0.0 | 5.88e-15 | 0.0 |
| 2 | 0.0 | 1.89e-15 | 0.0 |
| 3 | 0.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 0.0 |

A few ULP on a T scale of 0.74, in T only, nondeterministically. The test itself failed **4 of
6** runs. With `OPENBLAS_NUM_THREADS=1` the same code was **4/4 bit-exact**, because then the
racing save/restore is `1 -> 1 -> 1`.

**Why it never showed on CI.** The race needs BOTH `threadpoolctl` installed AND an environment
pool > 1. The 2-core runner and the WSL build (no `threadpoolctl` -> the cap is inert and both
paths run at the environment default) satisfy neither.

### 2.2 Fix

`lumenairy/elements/rcwa/stack.py`:

1. **Hoist the cap.** One `with _blas_threads_quiet(blas_per_worker), _blas_limit():` around the
   WHOLE dispatch, on the calling thread, wrapping both the serial loop and the thread pool. One
   enter/exit per sweep; every solve in both paths runs at the same BLAS thread count, so
   byte-identity holds **by construction** instead of being asserted.
2. **Clear the per-worker request.** `_solve_one` now runs under `_blas_threads_quiet(None)`, so
   `solve`'s own `@_with_blas_limit` finds nothing to apply and does not re-enter the global
   limiter. Without this the SERIAL branch -- which runs on the caller's thread, where the
   sweep-level request is set -- nested one enter/exit per wavelength inside the sweep-level one
   (measured: 7 applications for a 6-point serial sweep). Harmless while single-threaded, but it
   is the same pattern, and clearing it makes "exactly one cap application per sweep" a
   pinnable invariant.
3. **Corrected the three false thread-locality claims** listed in S2.1, each with the measured
   demonstration and the rule that follows ("apply the cap once around a parallel section, never
   once per worker").
4. **Corrected the method docstring's mechanism** for byte-identity.

### 2.3 Null control -- the fix moves NO numbers

sha256 of `(R, T, jones)` for the 12-point sweep:

| configuration | sha256 |
|---|---|
| post-fix, default pool (24), `max_workers=1` | `66eb265b1417aafb...42cfe2e` |
| post-fix, default pool (24), `max_workers=4` | `66eb265b1417aafb...42cfe2e` |
| `OPENBLAS_NUM_THREADS=1`, `max_workers=1` | `66eb265b1417aafb...42cfe2e` |
| `OPENBLAS_NUM_THREADS=1`, `max_workers=4` | `66eb265b1417aafb...42cfe2e` |

The `OPENBLAS_NUM_THREADS=1` rows are provably identical pre- and post-fix (the global is
always 1, so the old per-worker save/restore was a no-op). All four agree, so **the fix removed
the nondeterminism without moving a single bit**. Re-verified after change (2) as well.

A **fail-before switch is not applicable** and would be misleading: the prior behaviour was
nondeterministic, so there are no "prior bits" to reproduce. The fail-before evidence is the
measured pre-fix divergence table in S2.1, which the post-fix code cannot produce.

### 2.4 Test changes

* `test_threaded_sweep_is_byte_identical_to_serial` widened from a single 1-vs-4 comparison to
  `max_workers in {2, 4, 8}` (the pre-fix race fired nondeterministically, so one comparison
  reported green on ~1 run in 3 while the contract was broken), and asserts **tolerance at 0.0
  on the max absolute difference** per the standing rule, so a failure reports the magnitude.
  **Cost held at parity**: 12 wavelengths x 2 sweeps = 24 solves before, 6 x 4 = 24 solves now.
  That was not cosmetic -- see below.

### 2.4b Cost regression caught by the cross-build run -- fixed before shipping

The first version kept 12 wavelengths and added two worker counts: 48 solves instead of 24. On
Windows that is invisible (the cap is applied, every solve runs single-threaded BLAS). On the
**WSL/OpenBLAS build there is no `threadpoolctl`, so the cap is inert** and 8 concurrent
uncapped solves thrash a 24-thread pool: the file **did not finish inside 25 minutes** there,
against a repo that has already lost a release-verify shard to a 30-minute cap. Cutting the
wavelength count to 6 restores the original solve budget while still covering three worker
counts instead of one. **This is why the both-builds rule is not a formality**: the asymmetric
build is the one that priced the change.
* **NEW `test_threaded_sweep_applies_exactly_one_blas_cap`** -- the build-INDEPENDENT pin. The
  race cannot reproduce on CI, so a byte-identity assertion alone is green there whether or not
  the bug is present; that is how it survived. This test counts cap applications by substituting
  a counting stand-in for `_get_blas_controller`, which needs no `threadpoolctl` to be installed
  -- it runs on every build -- and asserts exactly 1 for `max_workers in {1, 2, 4}`.

### 2.4a A defect in that new test, found by M2 under `pytest -n 6` -- BAD TEST, fixed

The first version of the counting stand-in returned `contextlib.nullcontext()` instead of
delegating to the real controller. **It therefore switched off the very cap it was measuring.**
Measured directly, instrumenting the BLAS pool seen inside each `solve` during the sweep **[M]**:

| stand-in | BLAS threads inside each solve (4 solves, `max_workers=4`) |
|---|---|
| null-context (first version) | **24, 24, 24, 24** |
| delegating (fixed) | **1, 1, 1, 1** |

So the test ran 4 solver threads x 24 OpenBLAS threads = **96 threads against a library built
`MAX_THREADS=24`**. Alone, at `-n 2` and at `-n 4` that survives; under `pytest -n 6`, with six
such worker processes on the box, it took an xdist worker down --
`[gw2] node down: Not properly terminated`, reported as a failure of this test. **Not a
cross-worker counter race** (xdist workers are separate processes and the stand-in is
process-local); a resource kill caused by the instrument.

Fix: the stand-in **delegates** to the real controller when there is one, and falls back to a
null context only when there is none -- in which case the process was never capped anyway, so no
oversubscription is introduced. The test's stack was also cut to `n_orders=1` x 3 wavelengths: it
asserts a control-flow invariant, not a physical result, and had no business running a
physics-scale solve. Verified: passes alone and under `-n 6`.

**Lesson:** an instrument must not disable the mechanism it instruments. The counting stub was
"harmless" by construction -- it changed no numbers -- but it silently removed a resource guard,
and the failure surfaced two orders of parallelism away from the cause.

### 2.5 Evidence

* Post-fix: **10/10 bit-exact** at 4 and 8 workers (vs ~50% pre-fix); pool correctly restored to
  24 after a sweep.
* `tests/unit/test_v5_20_8_rcwa_threaded_sweep.py`: 4 passed, Windows; repeated runs green.
* Cross-build: WSL/OpenBLAS -- see S4a.
* Neighbouring BLAS suites green: `test_audit_s5_8_perf_noloss.py` (the cached
  `ThreadpoolController` win) and `test_niche_audit_m4_m5_m6_rcwa.py` (audit M6, the inert-cap
  warning) -- both builds.
* Ruff clean.

### 2.6 The same defect exists on three PMM sites -- referred

The identical per-worker-cap pattern is present at `lumenairy/elements/pmm/stack.py:2867` and
`:3037` (`PMMStack.solve_vs_wavelength`) and `lumenairy/elements/pmm/stack2d.py:1142`, each with
its own byte-identity pin (`tests/unit/test_v5_21_pmm_threaded_sweep.py`). Those files are M2's;
**the fix transfers verbatim** (hoist the `with` around the dispatch, clear the request inside
the per-wavelength function, and add the one-cap counting test). Referred to M2/M3.

---

## 3. D-1 -- the eight staleness corrections

Verified against the working tree at `d30f1ca` + M2's uncommitted edits. **Two of the eight had
already been fixed by concurrent missions**, which is itself the finding -- the plan's S-5 and
S-6 are no longer live defects and their entries had to be rewritten as *shipped*, not restated.

| id | verdict vs current tree | landed in |
|---|---|---|
| S-1 | **TRUE** (line numbers off by 1-2; the `J` assignment is split across two lines) | roadmap S1.2 items 1-2 |
| S-2 | **TRUE** (signature spans `stack2d_pure.py:123-124`, and carries one extra `degree=None` alias) | roadmap S1.2 item 5 |
| S-3 | **TRUE** (`sla.eig` at `twod_staggered.py:722`; `dim` is an attribute at `:249`, not a property) | roadmap S1.3 |
| S-4 | **TRUE in substance, all five citations STALE** (M2 restructured them) | impl report S2; parent audit R-6 |
| S-5 | **ALREADY FIXED** -- N-4's helper `_perlayer_window_grids` has landed; zero verbatim copies remain | roadmap T3-1 row |
| S-6 | **ALREADY FIXED** -- M1's T3-3 clamp has landed; "latent only" was itself refuted | roadmap T3-3 row |
| S-7 | **TRUE**, but the stale "Opt-in" comment is at `rcwa/_core.py` ~1828 and sits *before* the resolver, not after -- the plan's `1483-1485` is a different function | roadmap S6; `rcwa/_core.py` |
| S-8 | **TRUE** (a+b); cache is now at `rcwa/_core.py:3559-3571`, not `3165-3222` | roadmap S6 |

### Where each correction landed

**`docs/audits/ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md`**

* **S-1** (S1.2 items 1-2): the 2-D pure stack uses the Granet-2023 staggered *modified-Legendre*
  basis with **uniform** segments (`twod_staggered.py:148`, `:174` with the in-code comment
  `# segment boundaries on the eps walls (Eq.31, uniform)`, scalar `J` at `:171-172` used at
  `:261-266, 341-346, 566-568, 641-643`); `_lagrange_eval` is 1-D-only (`_core.py:3609`) with
  zero occurrences in the 2-D files. Consequences recorded: there are no per-layer *wall
  positions* in 2-D; the 1-D wall-collision pathology cannot occur; the real blocker is wall
  representability on a uniform lattice (a 2 deg taper at ns=6 needs `Nx ~ 390`, `eigdim = 1.5e7`
  -- impossible), so **a mortar alone does not unlock 2-D tapers**; N-1 is the enabling change.
  Also added: the dense 2-D cross-mass is a rejected design (Kronecker form required).
* **S-2** (S1.2 item 5): the pure 2-D stack has **no fold at all** -- struck, with the greps and
  the corrected reading (the risk is void; the real item is a new perf prerequisite, N-5).
* **S-3** (S1.3): the missing cost/memory model added as a table, with the `2q^2` pencil chain,
  plus the two omitted facts (`Nx == Ny` is required; `PMM2DStack` is a transitional alias to the
  hybrid, so S1 carries an API cutover).
* **S-5** (T3-1 row): "one-line change to the window loop" was false when written (five verbatim
  copies) and is **true now** -- N-4's helper, the `window_halfwidth` knob and the corrected
  ~4.6x-on-the-whole-solve cost recorded.
* **S-5 cont.** (T3-2 row): "both levers unexplored" is stale; the +-2 window is measured and
  recorded at `_core.py:3551-3560`. Added the "quote a band, don't force convergence" and
  "ER alone is rejected" rules.
* **S-6** (T3-3 row): struck and rewritten as **shipped**, with the full silent-rank-deficiency
  mechanism, the sibling comparison, and M1's fix + fail-before switch.
* **S-7** (S6 bullet): the `'li'` fold is in `rcwa_efficiency_2d` / `PreparedRCWA2D` /
  `RCWAStack._layer_even_spec`, **not** `rcwa_jones_2d` (whose gate demands `laurent`); the real
  gate is `li_ops is not None`, and a uniform `'li'` cell reroutes to `'laurent'`; "ON by default"
  excludes `rcwa_efficiency_2d_shapes` (`symmetry=False`) and the entire 1-D core (no `symmetry`
  kwarg anywhere).
* **S-8a** (S6 bullet): the cache is consulted at exactly two sites, both `RCWAStack`
  half-spaces, non-traced only; uniform **interior** layers and every `twod.py`/`oned.py`/Berreman
  site call the uncached form. **Any campaign claim of a cache-hit win on many-uniform-layer
  stacks is not yet realised.** Note the **source was already correct** here and needed no fix:
  the cache's own comment says "Cache of homogeneous **half-space** eigenmodes" and "one
  `RCWAStack.solve` touches exactly 2 entries (sup + sub)". Only the roadmap over-read it. Making
  uniform *interior* layers hit the cache is a real perf item (LEV-4 completion) but it is a
  behaviour change with a bit-identity question attached, so it is NOT folded into a doc mission.
* **S-8b** (S6 bullet): Li-2003 `L2.L1` is wired into `rcwa_jones_2d` only; the scalar pixel
  engine still builds the Schuster NV field (the code R-4 is about); the PMM port is
  **separable-only** and raises on a crossed cell.

**`lumenairy/elements/rcwa/_core.py`** (S-7c, in-code): the LEV-3 comment block preceding
`_symmetry_on` said `Opt-in (symmetry=True)` five lines above a docstring saying `"auto"` is the
default -- a direct self-contradiction on the shared RCWA path. Rewritten to "ON by default
(`symmetry='auto'`) since v5.21", with the opt-OUT rationale kept (the fold moves the result at
~1e-12, so `False` is bit-identical to the pre-fold path) and the non-universality spelled out.

**`docs/audits/AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md`** (S-4): the "inert by construction
(there is no global union to snap)" headline struck and replaced with the measured contract --
the window IS a `2*halfwidth+1`-layer union, `_perlayer_window_grids` passes
`min_feature / period` straight into `_pmm_union_grid` (`_core.py:3599-3606`) and the snap fires
whenever `min_feature > 1e-9` (`_core.py:3448, 3460`). Dormant at the library default (7 pm),
**live above it**, and it merges exactly the adjacent-slice pairs that matter (5.41 nm offset vs
a 5.00 nm coat). A user carrying the shared path's recommended 1.5 nm gets walls moved by up to
0.75 nm on a path documented as immune -- a ~16% ER change by the parent audit's own measurement.
Correct statement: *dormant at the library default, and the intended lever for T3-2.*

**`docs/audits/AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md`** (S-4): R-6's "retires
`min_feature` as an accuracy knob" corrected in place.

---

## 4. What this mission did NOT do, and why

| item | status | reason / owner |
|---|---|---|
| **T3-7** lattice wall quantisation | **NOT STARTED** | It is a change to `_pmm_union_grid` in `lumenairy/elements/pmm/_core.py` -- the exact function the concurrent M2 mission has restructured this session (uncommitted). Editing it would conflict with a live agent, and this mission was instructed not to edit PMM code. Design is unchanged from the plan; the sizing rule (Delta set by the separations to REMOVE, ~1 nm class, not the features to keep -- finer lattices merge FEWER pairs) must be carried into the docstring verbatim. |
| **T3-5** taper-aware `min_feature` warning | **NOT STARTED** | Same reason: `self._taper_recipes` and the `min_feature` default live in `lumenairy/elements/pmm/stack.py`. Note the constraint that survives: **warning + documented recommended value, NOT a default flip**, and the geometry-built route records no recipe, so its heuristic needs a labelled known-vertical / known-sheared / known-tapered set before it may speak (R-1b: a false pathology claim is worse than silence). |
| **S-4 in PMM source** | **DONE** (split M2 / M4) | Four repeats of the immunity claim existed in `lumenairy/elements/pmm/`. M2 corrected three: the `_core` window-grid contract, the `PMMStack` constructor comment, and the `_solve_vertical_perlayer` docstring (`"min_feature never enters"`). The fourth -- the `stabilize='slices'` **raise text shown to users** (`per-layer grids have no cross-layer walls to perturb`) -- was still live after M2 completed and was corrected by **M4**. The refusal itself is correct and unchanged; only its stated reason was wrong. Corrected reason: the consensus probes perturb the ONE shared union grid and read the spread across those probes, and there is no shared union grid on this path. A repo-wide grep for the three immunity phrasings now returns only the campaign plan and the correction notes themselves. |
| **CHANGELOG.md v5.32.0** | **NOT EDITED** (instructed) | The v5.32.0 PMM block carries `(inert on this path)`. It is a shipped release note, and this mission was instructed not to touch `CHANGELOG.md`. Whoever writes the 5.33.0 entry should carry the S-4 correction forward there. |
| ~~`_core.py:3535` dangling citation~~ | **RESOLVED during this mission** | M2's new comment cited `docs/audits/PMM_M2_WINDOW_CONTRACT_2026_08_04.md` before that file existed; it has since landed. Flagged here only because a citation that lands *after* the code is a race worth noticing -- re-check before the release tag. |
| **F-1's underlying code defect** | **REFERRED to M1/M2** | S1.5. `pmm/twod_jones.py`'s NumPy in-plane route manufactures ~1.8% energy on a lossless cell at 24 BLAS threads. Same class as M1's census, one file outside its scope. |
| **PMM's copy of the F-2 bug** | **REFERRED to M2/M3** | S2.6. Three sites, fix transfers verbatim. |

---

## 4a. Cross-build evidence (the "S8" referenced above)

| suite | Windows (py3.14.6, numpy 2.4.4, scipy-openblas 0.3.31, pool 24, `threadpoolctl` present) | WSL (py3.12.3, numpy 2.4.6, OpenBLAS, **no** `threadpoolctl`) |
|---|---|---|
| `test_audit_s5_8_perf_noloss.py` | 5 passed | **34 passed** combined with the row below |
| `test_niche_audit_m4_m5_m6_rcwa.py` | 28 passed, **1 skipped** (`threadpoolctl installed: the cap is effective here`) | that same test **RUNS** here (the inert-cap branch) -- complementary, not duplicated coverage |
| `test_v5_20_8_rcwa_threaded_sweep.py` | 4 passed (also 4 passed under `-n 6`) | **NOT COMPLETED -- see the caveat below** |
| `test_v5_20_2_pmm_jones_2d_jax.py` | 14 passed (leg 2 ACTIVE) | **NOT COMPLETED**; leg 2 would be skipped by construction (`_blas_pin` -> `None`) |
| combined RCWA/BLAS run | **37 passed, 1 skipped** | 34 passed on the two BLAS-machinery suites |
| `ruff check lumenairy/ tests/unit/` | -- | **All checks passed!** |

**CAVEAT, stated rather than papered over.** The two solve-heavy suites did not complete on the
WSL build within a 25-minute cap during this mission, because the Windows host was saturated by
three concurrent campaign missions and WSL shares those cores -- calibration measured a single
sweep solve at >75 s there against ~4 s on the idle Windows path, a ~20x load penalty, and even a
single 6-wavelength serial sweep did not finish in 15 minutes. **This is environmental, not a
property of the change**: on a build without `threadpoolctl`, `_blas_limit` returns a null
context on BOTH the pre-fix and post-fix code paths, so the WSL BLAS behaviour is *identical
before and after* -- the fix is a no-op there by construction, and the only WSL-visible delta is
the test file itself, whose solve budget is held at parity (S2.4b). **These two suites must be
re-run on WSL on an idle box before the release tag.**

The two builds are deliberately ASYMMETRIC in the property that matters here: Windows has
`threadpoolctl` and a 24-thread pool (the only configuration in which either bug can fire), WSL
has neither. So WSL is the null-environment control -- it exercises the
`threadpoolctl`-absent branches of both fixes (`_blas_pin` returning `None`; `_blas_limit`
returning a null context in both sweep paths) and must be unaffected.

---

## 5. Reproduction commands

```
# F-1: the BLAS-thread dependence (vary only the env var)
OPENBLAS_NUM_THREADS=1  python -m pytest "tests/unit/test_v5_20_2_pmm_jones_2d_jax.py::test_pmm_jones_2d_jax_forward_matches_numpy[inplane]"
OPENBLAS_NUM_THREADS=24 python -m pytest "tests/unit/test_v5_20_2_pmm_jones_2d_jax.py::test_pmm_jones_2d_jax_forward_matches_numpy[inplane]"

# F-1's referred code defect: lossless closure of the NumPy solve alone
#   deg 11, 24 threads -> 2.0307687 (defect 3.1%);  1 thread -> 2.0125077 (defect 1.25%)

# F-2: the race (pre-fix), ~50-70% of runs on a >1-thread pool with threadpoolctl
python -m pytest tests/unit/test_v5_20_8_rcwa_threaded_sweep.py::test_threaded_sweep_is_byte_identical_to_serial

# F-2: the process-global demonstration
python -c "import threading,time,numpy;from threadpoolctl import threadpool_limits,threadpool_info; ..."
```

---

## 6. Acceptance gates (plan S4, M4 row)

| axis | gate | result |
|---|---|---|
| accuracy | no pinned number moves except where era-pinned | **PASS** -- F-2 sha256-identical (4 configurations); F-1 keeps every invariant bar at its historical value and era-pins the one that moved, verbatim, in source |
| null control | a control that must be unaffected | **PASS** -- F-1: the `[oop]` arm, <= 3.6e-14 at 1 and 24 threads; F-2: the `OPENBLAS_NUM_THREADS=1` configuration, provably identical pre/post |
| both-builds | Windows/OpenBLAS-24 + WSL/OpenBLAS | **PARTIAL** -- Windows complete (incl. `-n 6`); WSL green on ruff + the two BLAS-machinery suites (34 passed, and it covers the `threadpoolctl`-absent branch Windows skips), but the two solve-heavy suites could not complete under the concurrent-mission load. Structurally a no-op on that build (S4a). **Re-run required on an idle box before the release tag.** |
| default | no default changed | **PASS** -- `blas_per_worker=1` and `max_workers=None` unchanged; `min_feature`'s default untouched |
| docs | every S1.2 correction landed with its `file:line` | **PASS** for S-1..S-3, S-5..S-8 and the doc half of S-4; the three PMM-source repeats of S-4 are referred (S4 above) |
| ruff | clean | **PASS** (WSL, `lumenairy/` + `tests/unit/`) |

---

## 7. Lessons worth carrying

1. **"Thread-local" must be verified at the layer that APPLIES the setting, not the layer that
   records it.** Three source comments asserted thread-locality of a cap whose application is a
   process-global C call. The request was thread-local; nobody checked the other half.
2. **A green test on CI is not evidence the contract holds** when the failure mode needs
   hardware CI does not have. The byte-identity pin was structurally incapable of failing on a
   2-core runner. The durable fix was to pin the *mechanism* (one cap application per sweep),
   which is build-independent, next to the *symptom*.
3. **Before re-tuning a tolerance, vary one environment knob and re-measure.** Two releases spent
   re-tuning `_PAR_TOTAL`; one `OPENBLAS_NUM_THREADS` sweep showed the quantity was set by the
   thread count and the constant could never be right on both runners.
4. **A leaking process-global setting makes unrelated tests pass.** The reason F-1 "passed with
   its neighbour" was that F-2's bug left the pool at 1. Two independent-looking failures, one
   mechanism -- and diagnosing either one alone was impossible.
5. **Energy closure is the instrument that says WHICH engine moved.** The twin-vs-NumPy gap alone
   is symmetric and uninformative; each engine's own departure from a lossless budget is not.

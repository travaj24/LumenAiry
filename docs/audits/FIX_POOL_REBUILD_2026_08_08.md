# FIX -- the Newton pool worker was still REBUILDING the fit

**2026-08-08.  Branch `feat/pmm-per-layer-roadmap`.  Closes the one CI failure
that survived `FIX_CI_POOL_2026_08_06.md`:
`test_niche_newton_pool_both_fits.py::test_pool_result_is_bit_identical_to_
serial[polynomial]`, still failing at `max|delta| = 1.341e-11 / 1.358e-11` on
all four python lanes of shard 1/3 AFTER the Chebyshev backend pin shipped.  No
git command was run; `CHANGELOG.md` was not touched.**

---

## 0. VERDICT

> **The backend pin made the two sides evaluate in the same ORDER.  It did not
> stop them computing two different POLYNOMIALS.**
>
> `_newton_invert_chunk` rebuilt `_Cheb2DEvaluator` from the pickled grids,
> and building one RUNS `_solve_lstsq_thread_safe`: `G = A^T A` and `A^T b`
> over a ~78 000-row design matrix.  That is a BLAS reduction, and OpenBLAS
> reduces it in a THREAD-COUNT-DEPENDENT order.  A spawn worker does not
> inherit its parent's BLAS width -- `threadpoolctl`'s cap is process-global on
> OpenBLAS, so a long-lived pytest parent that has passed through a capped
> section is not at the environment default a fresh interpreter starts at -- so
> the two recovered coefficients differing in the last bits from byte-identical
> data.  MEASURED `max|dc|` 4.6e-15, which the Newton loop's `res < tol`
> threshold amplifies to **1.370e-11** of the field: CI's number.
>
> The parent now SHIPS its built fit (28 coefficients + two 28-entry index
> vectors per evaluator, ~2 kB against the ~1.9 MB of grids the payload already
> carries) and the worker EVALUATES it.  There is no second least-squares solve
> left to agree with, so `pool == serial` holds by CONSTRUCTION, at every BLAS
> width, on every build.

Nothing about the answer moves: the shipped coefficients are the ones the
SERIAL closure evaluates, so the pool now returns what the serial path returns
by definition rather than by two solves coinciding.

---

## 1. WHY THE BACKEND PIN WAS NECESSARY AND NOT SUFFICIENT

`FIX_CI_POOL` sec 1 was right about its own defect and right that it was not
the only one -- its sec 4.2 ledgered this exact residue:

> "**The lstsq fit is still assumed reproducible across parent and worker.**
> `_solve_lstsq_thread_safe` runs normal equations through BLAS, whose
> reduction order is thread-count dependent; a spawn worker inherits the same
> environment and CPU count as its parent, so the two agree in every
> configuration measured here ... A lane that pinned BLAS threads in the parent
> only -- e.g. via `threadpoolctl` at runtime rather than the workflow `env:`
> block -- would re-open the same CLASS of defect through a different door.
> The `unit` matrix deliberately does not pin (see `tests/conftest.py`), so this
> is not live today."

It was live.  The premise "a spawn worker inherits the same environment and CPU
count as its parent" does not carry to the BLAS WIDTH, because the width is not
environment alone: `threadpoolctl` sets it by calling
`openblas_set_num_threads()`, which is **process-global** (this library records
that itself -- see the M4 block at `elements/rcwa/_core.py:126` and
`elements/pmm/stack.py:2940`).  A parent that has been through any capped
section, or a `_blas_threads_quiet` window whose restore lost a race, runs at a
width its freshly spawned workers -- which always start at the environment
default -- do not share.

### 1.1 The asymmetry the CI log shows

| lane | `[polynomial]` | `[spline]` |
|---|---|---|
| py3.10 shard 1/3 | **FAIL 1.358e-11** | (ran in shard 3) PASS |
| py3.11 shard 1/3 | **FAIL 1.358e-11** | (shard 3) PASS |
| py3.12 shard 1/3 | **FAIL 1.341e-11** | (shard 3) PASS |
| py3.13 shard 1/3 | **FAIL** | (shard 3) PASS |

and NO `NewtonWorkerBackendUnavailable` warning anywhere in the log -- so the
backend pin was being honoured and the parent and its workers agreed on the
evaluator branch.  The split is the same one `FIX_CI_POOL` sec 1.1 named, read
one layer deeper: the spline worker rebuilds a `RectBivariateSpline`, which is
FITPACK -- single-threaded, no BLAS, no reduction-order axis.  The polynomial
worker rebuilds a least-squares fit.  **The polynomial fit is the only BLAS on
that code path**, and it is the only parametrisation that failed.

---

## 2. THE MECHANISM, MEASURED

Box: Windows 11 / py3.14.6 / numpy 2.4.4 / scipy-openblas 0.3.31.188.0
(DYNAMIC_ARCH, Haswell kernel, MAX_THREADS 24) / 24 cores, and WSL Ubuntu /
py3.12.3 / numpy 2.4.6 on the same tree.  Payload: the REAL one, captured from
this contract's own call (`N=1024`, `ray_subsample=2` -> 262 144 Newton points,
`newton_fit='polynomial'`, order 6, 77 841-point ray-fit grid, 4 chunks) by
substituting an in-process executor for `_get_persistent_worker_pool`.

### 2.1 The fit rebuild is not bit-stable across BLAS width

Coefficients of the three rebuilt evaluators, at BLAS width `W`, against the
same rebuild at the process default (24).  `scratchpad/repro_fit_rebuild.py`:

| W | bit-identical | `max abs dcoef` (Windows) | `max abs dcoef` (WSL) |
|---|---|---|---|
| 1 | **no** | 4.596e-15 | 4.596e-15 |
| 2 | **no** | 4.243e-15 | 4.243e-15 |
| 4 | **no** | 5.484e-15 | 5.484e-15 |
| 8 | **no** | 8.766e-16 | 8.766e-16 |

Two controls say this is the WIDTH and not noise
(`scratchpad/probe_fit_determinism.py`):

| control | result |
|---|---|
| 30 rebuilds in one process at ONE width | 30/30 bit-identical |
| parent vs a REAL spawned worker, both at the default width | bit-identical, 0.000e+00 |
| parent under `threadpool_limits(1)` vs a REAL spawned worker at the default width | **DIFFER, 2.537e-17** |

The second row is why this passes on a dev box and on Windows: same regime,
same bits.  The third row is the CI condition, reproduced through a genuine
`spawn` child rather than an emulation, and it is asserted in the suite by
`test_a_real_spawned_worker_uses_the_parents_fit`.

### 2.2 ...and that is exactly CI's number

`scratchpad/repro_pool_thread_split.py` runs the contract test's own shape --
serial vs 4 chunks through the REAL `_newton_invert_chunk` -- with the WORKER
body, and only the worker body, wrapped in a BLAS cap the parent does not have.
`max|dfield|` against serial:

| worker BLAS width | PRE-FIX | POST-FIX |
|---|---|---|
| 1 | DIFFER, 1.830e-12 | identical, `0.000e+00` |
| 2 | DIFFER, 1.062e-11 | identical, `0.000e+00` |
| 4 | **DIFFER, 1.370e-11** | identical, `0.000e+00` |
| 8 | DIFFER, 1.198e-12 | identical, `0.000e+00` |
| real spawn pool, worker regime = parent regime | identical, `0.000e+00` | identical, `0.000e+00` |

**CI reported 1.341e-11 (py3.12) and 1.358e-11 (py3.10 / 3.11).  The width-4
emulation gives 1.370e-11.**  Unlike `FIX_CI_POOL` sec 1.3 -- which reproduced
the mechanism at 5.167e-14 and explicitly declined to claim the numbers should
match -- this one lands on CI's value to two significant figures, on a
different OS and a different python, which is what makes it an identification
rather than a plausible story.

### 2.3 Why 1e-15 of coefficient becomes 1e-11 of field

Three steps, each independently measurable:

1. **Coefficients.**  `max abs dc` ~ 5e-15 on coefficients of order 1e-3 to
   1e+1, i.e. ~1e-15 relative -- a reduction-order difference and nothing more.
2. **Newton.**  The loop retires a point on `res < tol`.  A perturbation at the
   last bit moves a handful of the 262 144 points across that line, so they
   take one more (or one fewer) Newton step.  Measured worker-OPL deltas, same
   payload, worker at width W vs the parent's:
   `3.595e-18 / 6.912e-18 / 8.155e-18 / 1.220e-18` m at W = 1 / 2 / 4 / 8.
3. **Phase.**  The field is `|E| exp(i 2 pi OPL / lambda)`, and
   `2 pi / 1.31 um = 4.796e6 rad/m`, so 3e-18 m of OPL is 1.4e-11 of a
   unit-amplitude field.  `8.155e-18 * 4.796e6 = 3.9e-11` is the bound; the
   observed 1.370e-11 is that bound sampled where `|E|` actually is.

So the chain is: BLAS reduction order -> last-bit coefficients -> a few
points' Newton iteration count -> ~1e-18 m of OPL -> ~1e-11 of field.  Nothing
in it is a bug in anybody's arithmetic, which is precisely why it cannot be
fixed by making the arithmetic better.

---

## 3. WHAT SHIPPED

All in `lumenairy/elements/_lens_traced.py`.

```
_validated_cheb_backend(backend)            one gate, two callers
_cheb_fit_state(ev)                         built fit -> picklable dict
_cheb_fit_payload(Sx, Sy, So, newton_fit)   the three of them, or None
_Cheb2DEvaluator.from_state(state, ...)     construct WITHOUT fitting
```

* **Parent.**  `_invert_newton_parallel` sets
  `_spline_data['cheb_fit'] = _cheb_fit_payload(Sx, Sy, So, newton_fit)` at the
  DISPATCH site, immediately after the `cheb_backend` pin and before
  `args_list` is built -- for the same two reasons that site was chosen for the
  backend: only a dispatch needs it, and it must describe the state in force at
  dispatch time.  `Sx`/`Sy`/`So` there are the very objects the SERIAL closure
  evaluates, which is what turns "pool == serial" from a claim about two
  least-squares solves into a statement about one set of coefficients.
* **Worker.**  `_newton_invert_chunk` reads `cheb_fit` and builds through
  `_Cheb2DEvaluator.from_state`, which performs no arithmetic at all.  A
  payload with NO key keeps the historical rebuild, mirroring the
  `get('newton_fit', 'spline')`, `get('newton_max_iters', ...)` and
  `get('cheb_backend', None)` tolerances beside it.
* **`from_state` VALIDATES.**  A state whose `coeffs`, `K1`, `K2` and
  multi-index list do not all agree in length raises `ValueError` rather than
  broadcasting its way to a plausible-looking field.  Silence there would be
  the same class of defect this file is about.
* **The backend pin is UNTOUCHED and still needed.**  `ev_value_and_grad` still
  has two implementations, the payload still pins which one, and a worker that
  cannot honour a pinned `'numba'` still raises
  `NewtonWorkerBackendUnavailable`, still falls back to the serial closure, and
  still latches.  Shipping the fit fixes WHICH POLYNOMIAL; the pin fixes IN
  WHAT ORDER it is summed.  Both are required.
* **The memory model is UNTOUCHED.**  `_newton_worker_bytes`,
  `_newton_resolve_workers`, `on_pool_memory` and the unguarded-`__main__` rule
  are byte-for-byte as they were.  The grids stay in the payload (the spline
  path still needs them and `_fit_points` still prices exactly one copy), so
  the clamp cannot move; the fit adds ~2 kB per payload, measured at 0.3% of a
  single grid.

### 3.1 Why not the alternative (pin BLAS to one thread around the fit)

The task offered it, and it was measured before it was rejected:

1. **It does not hold.**  A single-thread cap makes the fit reproducible only
   if BOTH sides take it.  `threadpool_limits` on OpenBLAS is process-global
   (this library documents that, twice), so applying it inside a library call
   that can be reached from `parallel_amp`'s ThreadPoolExecutor changes the
   width under sibling threads -- the M4 race, re-introduced on the traced-lens
   path.
2. **It costs, on the path that is already the fast one.**  The fit is
   77 841 x 28; capping it to one thread taxes every `apply_real_lens_traced`
   call, pooled or not, to fix a defect that only exists in a pooled worker.
3. **It fixes one door.**  Width is the axis measured here, but any difference
   between two builds' reductions (a different OpenBLAS kernel selected by
   DYNAMIC_ARCH on heterogeneous runners, an MKL parent against an OpenBLAS
   worker) re-opens it.  Shipping the coefficients closes the CLASS: the worker
   performs no reduction, so it has none to disagree about.

Table 2.1's W=1 row is the direct evidence for (1): a fit built at width 1 is
NOT the fit built at width 24, so a cap in the worker alone -- which is all a
worker can apply to itself -- makes the disagreement worse, not better.

### 3.2 Why the spline path is left alone

`_cheb_fit_payload` returns `None` for `newton_fit='spline'`, and that worker
still rebuilds through FITPACK.  `RectBivariateSpline`'s `regrid` is
single-threaded and takes no BLAS path, so it has no reduction-order axis; the
CI evidence agrees, `[spline]` having passed in every lane it ran in while
`[polynomial]` failed in all four.  Shipping a `tck` would be motion without a
measurement behind it.

---

## 4. TESTS

### 4.1 New (`tests/unit/test_fix_newton_pool_memory.py`, 40 -> 48)

| test | what it pins |
|---|---|
| `test_a_rebuilt_fit_is_not_blas_width_stable` | the PREMISE and the FAIL-BEFORE: coefficients differ across BLAS widths, are stable at a FIXED width, and differ by a reduction-order amount rather than a wrong-answer amount.  Skips, with the reason, on a build whose BLAS cannot be capped or does not thread this shape |
| `test_the_worker_evaluates_the_shipped_fit_and_never_re_fits` | the worker reaches `_solve_lstsq_thread_safe` ZERO times on a shipped-fit payload, and still exactly three times on a keyless one |
| `test_the_shipped_fit_is_evaluated_identically_at_any_blas_width` | THE CONTRACT, unconditional: the same worker call at widths 1 / 2 / 4 / default returns bit-identical OPL |
| `test_from_state_reproduces_the_parents_evaluator_bitwise` | state -> pickle -> evaluator evaluates byte-identically to the object it came from, runs no solve, and validates its backend |
| `test_the_shipped_state_carries_the_fit_and_not_the_grids` | exactly nine keys, consistent shapes, < 16 kB and < 1/20 of ONE grid, `None` for spline |
| `test_a_truncated_fit_state_is_refused_not_broadcast` | a state short one coefficient raises instead of broadcasting |
| `test_the_dispatch_path_ships_the_built_fit` | wiring: the payload line, its position before chunking, the worker's read, `from_state`, and that the keyless rebuild path still exists |
| `test_a_real_spawned_worker_uses_the_parents_fit` | the probe: a REAL `spawn` child, parent fit under a cap the child does not inherit, worker holds the parent's coefficients bit for bit and ran 0 solves.  Prints the child's own re-fit delta as the fail-before evidence (measured 2.537e-17 here) |

### 4.2 Changed

* `test_niche_newton_pool_both_fits.py::test_worker_payload_carries_what_the_
  polynomial_fit_needs` -- also requires the payload to carry `cheb_fit`, the
  worker to read it, and the worker to construct via `from_state`.
* `test_niche_newton_pool_both_fits.py` module docstring -- it asserted "the
  Chebyshev fit is a deterministic lstsq on identical data so every worker
  recovers the same coefficients".  That sentence was the defect, stated as the
  safety argument.  Replaced with the measurement.
* `test_pool_result_is_bit_identical_to_serial` is unchanged: it was already
  unconditional, and it is the test this fix exists to make true.

### 4.3 Green matrix

Both pool files (`test_niche_newton_pool_both_fits.py` 23 +
`test_fix_newton_pool_memory.py` 48 = 71), two mounts, three BLAS widths, no
skips in any cell:

| mount | `OPENBLAS_NUM_THREADS` | result |
|---|---|---|
| Windows 11 / py3.14.6 | 1 | **71 passed** (101.9 s) |
| Windows 11 / py3.14.6 | 2 | **71 passed** (98.2 s) |
| Windows 11 / py3.14.6 | default (24) | **71 passed** (102.0 s) |
| WSL Ubuntu / py3.12.3 | 1 | **71 passed** (107.5 s) |
| WSL Ubuntu / py3.12.3 | 2 | **71 passed** (99.6 s) |
| WSL Ubuntu / py3.12.3 | default (24) | **71 passed** (118.8 s) |

### 4.4 Regression sweep

Every suite in `tests/unit` that names `_Cheb2DEvaluator`,
`_newton_invert_chunk` or `newton_fit` -- selected by grep, not by memory --
run together on Windows at the default BLAS width:

| suite | result |
|---|---|
| `test_hammer_h3_traced_nyquist_guard` + `test_fix_d5_fit_domain_basis` + `test_niche_audit_e_prepared_and_enums` + `test_niche_audit_w3_elements` + `test_niche_c1_consolidation` + `c6` + `c8` + `c11` + `c12` + `d1` + `d7` + `d8` + `d9` | **364 passed, 0 skipped** (723.9 s) |
| `ruff check` (Windows, ruff via the project venv) on all three changed files | clean |

The C1 / C6 / C8 byte-identity niches are in that list on purpose: they are the
contracts that pin `_solve_lstsq_thread_safe`'s exact bits, and a change that
altered what the fit RETURNS -- rather than merely where it is computed --
would break them.  It does not: the parent's fit is built by the identical code
path it was before, and the worker now reads its output instead of recomputing
it.

### 4.5 A note on the version labels

The in-code tags read `v5.32.3` for the backend pin and `v5.33.1` for this fix,
which is not a typo.  `5.33.0` shipped between them (CHANGELOG 2026-08-06,
the design-121 capstone); `[Unreleased]` is empty as of this writing, so the
next patch is `5.33.1`.  Neither of these two fixes has a CHANGELOG entry yet
-- both are branch work.

---

## 5. WHAT THIS DOES NOT CLOSE

1. **Which shard-1 test leaves the parent at a non-default BLAS width is not
   identified.**  The mechanism is proved on a real spawned worker in the
   direction that matters (parent capped, worker not), and the fix removes the
   dependence entirely rather than repairing one direction of it, so nothing
   here rests on naming the polluter.  The candidates are the process-global
   `set_blas_threads` / `_blas_threads_quiet` paths this repo already documents
   (`rcwa/_core.py` M4, `pmm/stack.py`), and a threaded sweep whose restore
   loses a race is the obvious shape.  If it is ever worth chasing, the
   instrument is `threadpoolctl.threadpool_info()` in
   `conftest.describe_process_state`, which already dumps `OPENBLAS_*` env but
   not the LIVE pool width.
2. **The other consumers of `_solve_lstsq_thread_safe` are untouched.**  The
   residual-eikonal and decentred-fit solves have the same reduction-order
   property, but nothing re-runs them in a second process, so they have no
   cross-process identity contract to break.  This fix is about the pool.
3. **No speed claim.**  The worker does strictly less work (three ~78 000-row
   least-squares solves per chunk are gone), but the Newton loop dominates and
   no benchmark was run.  FIX_D1's measurements stand.
4. **`newton_fit='spline'` still rebuilds** -- sec 3.2, on evidence.

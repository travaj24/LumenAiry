# Release-gate closure for 5.33.0

**Session:** 2026-08-04 (into 2026-08-05) - **Branch:** `feat/pmm-per-layer-roadmap`
**Role:** gate-closure agent for the PMM per-layer campaign (M1-M5 all closed before this ran).
**Scope:** close the two open verification gates, root-cause and fix every named failure from the
breadth sweeps, and hand the orchestrator a complete uncommitted-file inventory.

Legend: **[M]** measured here, **[A]** read from the tree, **[R]** referred to another owner.

---

## 0. Summary

| # | item | verdict |
|---|---|---|
| 1 | narrow six broad `except`s in `propagators/carrier.py` | **CANCELLED mid-flight by the coordinator.** Work completed, then **reverted byte-exact**; the file is foreign territory (a live D8/D9 session owns it). Two real defects found before the retraction are **referred with reproductions** (S2) |
| 2 | `test_dual_annealing_terminates_quickly_when_cancelled` (WSL chunk 1) | **FIXED -- BAD CODE + a bad bar.** scipy polls the callback only on a new global best, so `progress.cancel()` was never observed: **0 polls / 404 evaluations = the full budget, on BOTH builds**. Objective-level poll added; now **1 poll / 1 evaluation** (S3) |
| 3a | PMM collateral set, both builds (M3's open gate) | **CLOSED.** Windows **1657 passed, 2 skipped, 0 failed**; WSL see S4.1 |
| 3b | M4's two solve-heavy suites on WSL (M4's PARTIAL gate) | **CLOSED. 18 passed** (S4.2) |
| 4 | breadth-sweep chunk collection + any new named failure | **2 new names, both root-caused and FIXED as BAD TESTS** -- and both were the same trap (S5) |

**The through-line of this session.** Four of the five failures examined were **absolute bars on
magnitudes that move with the BLAS thread count or the build**, and the one that was not
(dual_annealing) was *hiding behind* such a bar. That is the third, fourth and fifth instance of
the trap M4 named in its lesson 3 and M3 hit on `worst_rcond`. Every fix here replaces the
constant with a comparison measured in the same process.

**What was refuted along the way:**

* that the dual_annealing failure was a WSL/tolerance artifact -- it is a **real cancellation
  defect present on Windows too**, which the 30 s bar merely hid (S3.1);
* that the m1 conditioning failure was a **build** difference -- Windows and WSL agree to **ten
  digits** at the same thread count; only the thread count moves the answer (S5.1);
* that the w8 failure indicated a broken solve -- the **signal is build-independent to five
  digits** while only the sanity gate drifts 12x (S5.2);
* my own first two probes were wrong and were caught before use: one ran with the wrong
  wavelength and reported a perfectly converged answer (S5.1), and the poll counter was
  validated against a method with a working callback before any defect was claimed (S3.2).

---

## 1. Builds

| | Windows | WSL |
|---|---|---|
| Python | 3.14.6 | 3.12.3 |
| NumPy | 2.4.4 | 2.4.6 |
| SciPy | 1.17.1 | 1.17.1 |
| BLAS | scipy-openblas 0.3.31 (24-thread pool) | distro OpenBLAS, no `threadpoolctl` |

**Two WSL virtualenvs exist and were checked rather than assumed** [M]. The breadth sweep runs in
`~/lumen_venv`; the campaign's documented WSL build (M3 S9) is `~/lumvenv`. They are identical in
every version that matters -- py 3.12.3 / numpy 2.4.6 / scipy 1.17.1 / pytest 9.0.3 -- so the
verifications below, run in `~/lumvenv`, apply to the failures observed in `~/lumen_venv`.

Unless a table says otherwise, every number is with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Where the thread count is the variable
under study, it is stated per row.

---

## 2. Item 1 -- CANCELLED, reverted byte-exact, and two defects referred

### 2.1 Retraction status [A]

The coordinator retracted item 1 mid-flight: `lumenairy/propagators/carrier.py` is being edited
right now by the session that committed the D8 work, and is not to be touched.

**The instruction included a `git checkout -- lumenairy/propagators/carrier.py` as the revert, and
that would have destroyed the other session's work.** By the time the retraction arrived, that
file carried **257 insertions / 100 deletions**, the great majority of them a live "niche D9"
change (a chief-ray-centred retrace grid) that has nothing to do with this agent. A file-level
checkout discards uncommitted work unrecoverably. The revert was therefore done **hunk by hunk**
with the exact inverse of each edit.

**Verification that the revert is exact** [M]: the whole region from
`def _multi_capture_worker_state` to `def propagate_traced_carrier_chain_multi(` -- which contains
every line this agent touched -- was extracted from `git show HEAD:...` and from the working tree
and compared:

```
D8 worker/pool region identical to HEAD: True
```

and the repo-wide count the budget test uses is back to its HEAD value:

| | count |
|---|---|
| budget (`_NON_UI_EXCEPT_BUDGET`) | 48 |
| repo-wide non-`ui/` `except Exception` now | **54** |
| of which `propagators/carrier.py` | **6** |

The budget test is therefore **red exactly as it was at HEAD**, and no CHANGELOG, budget constant
or test was modified to hide it.

### 2.2 The two except-budget tests: KNOWN-RED, OWNED BY THE D8 SESSION [R]

`tests/unit/test_audit_except_budget.py::TestExceptExceptionBudget::test_non_ui_except_exception_within_budget`
and `::test_non_ui_count_substantially_below_pre_sweep` are **red at HEAD and stay red**, both
builds. The six clauses are `lumenairy/propagators/carrier.py`'s process-pool bootstrap, added by
commit `95a9849`; the bisect is in M3 S7.1b. Deriving the raisable tuples is that author's call.
**This is recorded as known-red-with-owner, not as a silent pass.**

For the record, the work was completed before the retraction and reverted, so the owner does not
have to start cold: five of the six are ordinary swallow-guards that narrow cleanly
(`ImportError` for the module-import guards; the `(ImportError, AttributeError, OSError)` tuple
`_multi_resolve_workers` already uses for the same `get_ram_budget` probe; `(ImportError,
TypeError, ValueError)` for the `set_max_ram` restore). **The sixth is different and is the one
worth flagging:** the `except Exception as exc` around the pool is a *re-raise wrapper* whose job
is to name the failing congruence, and a worker runs an entire traced chain, so it can fail with
any exception type in the library. No named tuple spans that, and a tuple that tried would
silently stop naming the congruence for every type it missed. The structural answer is
`Future.exception()`, which **returns** the worker's exception instead of raising it, so the wrap
stays fully general with nothing caught at all. That was implemented and verified green (25
passed, Windows) before the retraction.

### 2.3 REFERRAL 1 -- `congruence_workers > 1` does not work at all on Linux [M]

Found while verifying the (now reverted) narrowing; it is a property of the shipped D8 code, not
of any edit made here.

`ProcessPoolExecutor` is constructed with no `mp_context`, so it takes the platform default:
`spawn` on Windows, **`fork` on POSIX through CPython 3.13**. A congruence chain has already run
BLAS by the time the pool is built, and libgomp refuses to be forked. The child dies before
executing a line of the chain:

```
Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.
PARALLEL RAISED: RuntimeError ... Original error: BrokenProcessPool('A process in the
process pool was terminated abruptly while the future was running or pending.')
```

Reproduced from a standalone script (not pytest) on the WSL build; the serial leg of the same
fixture completes normally at 61.5 s. This is the mechanism behind two of the three WSL D8
failures in sweep chunk 4 (`test_parallel_recombined_field_is_fp_identical_to_serial`,
`test_parallel_preserves_per_congruence_records_in_order`), and it reproduces in isolation --
they are not a pollution artifact. CPython agrees the default is wrong: 3.12 emits a
`DeprecationWarning` for fork in a multi-threaded process and 3.14 stops defaulting to it on
Linux.

**Suggested fix (verified to be the right shape, not shipped):** pass
`mp_context=multiprocessing.get_context('spawn')`. Spawn is the semantics the niche was written
for -- a worker that imports the library fresh is the entire reason `_multi_capture_worker_state`
exists -- so pinning it makes every platform agree instead of only the one where the default
happened to match. Note it is a genuine behaviour change for Linux callers whose `__main__` is
not import-safe, which is precisely the condition `_multi_looks_like_spawn_bootstrap` already
detects and explains.

### 2.4 REFERRAL 2 -- the worker-state snapshot is unpicklable after any callable glass [M]

This explains the **third** WSL D8 failure, `test_snapshot_is_picklable`, which **passes in
isolation and fails in a full-suite chunk** -- and it is a production defect, not a test artifact.

`_multi_capture_worker_state()` copies `GLASS_REGISTRY` verbatim into the snapshot, and the
snapshot is handed to `ProcessPoolExecutor` as `initargs`, so it **must pickle**. Registering a
callable "model glass" is a fully supported and heavily used pattern -- **43 test files do it, many
at module level**, e.g. `GLASS_REGISTRY['_K1A'] = lambda wl: 1.5168`. Demonstrated read-only:

```
clean snapshot pickles: True
after a CALLABLE glass registration: PicklingError: Can't pickle <function <lambda> ...>
```

So any process that has imported one of those modules can no longer snapshot, and any user with a
callable glass gets a pickling failure at pool startup rather than a chain result. The capture
already filters flags to a scalar allow-list; the glass tables get no equivalent treatment.

---

## 3. Item 2 -- dual_annealing cancellation: BAD CODE, with a bar that hid it

### 3.1 Root cause [M]

scipy invokes the `dual_annealing` callback from exactly one place --
`EnergyState.update_best` (`scipy/optimize/_dual_annealing.py` lines 197-206) -- so it fires
**only when a new global best is found**; the initial best is assigned in `reset()` with no
callback at all. On a run that plateaus it never fires, so `_scipy_cb_da` -- the whole v4.13.2 C.3
fix -- was never reached and `progress.cancel()` was ignored for the entire budget.

Measured on the regression test's own fixture, progress pre-cancelled:

| build | `is_cancelled` polls | merit evaluations | wall |
|---|---|---|---|
| Windows py3.14.6 / scipy 1.17.1 | **0** | **404** | 47.6 s |
| WSL py3.12.3 / scipy 1.17.1 | **0** | **404** | **993.3 s** |

Uncancelled controls on the same fixture: `max_iter=1` costs 6 evaluations, `max_iter=2` costs 8.
So 404 **is** the full `max_iter=200` budget -- the cancellation changed nothing.

**The two builds agree exactly on the evaluation count and differ 21x on wall time.** That single
pair of facts is both the diagnosis and the argument for the new bar.

**Why the test passed on Windows.** The old assertion was `elapsed < 30 s`. In a full-suite run
the process is warm and 404 evaluations fit inside 30 s on the Windows box; on WSL the same 404
evaluations take 993 s. The bar was measuring the machine, and a genuinely broken feature sat
underneath it for the whole of v4.13.2 through v5.32.

### 3.2 The instrument was validated before the defect was claimed [M]

A poll counter that never registers would produce the same "0 polls" reading as a real defect. It
was therefore run against methods with working callbacks first: `differential_evolution` registers
**1** poll through the identical patch. The counter works; the zero is real. (The stand-in also
delegates to the real `is_cancelled` rather than answering on its own -- M4's lesson that an
instrument must not switch off what it measures.)

### 3.3 Fix -- `lumenairy/optimize/driver.py` [M]

The callback is **kept** (it is the cheaper exit whenever a best *is* found) and the poll is
**also** made in the objective, which every scipy method calls on every evaluation and which no
plateau can skip. A private `_DualAnnealingCancelled` is raised from the objective and caught at
the dispatch site, which returns the best point seen as a normal `DesignResult` with
`success=False` -- a user stop is not an error, and every other method's cancellation path returns
a result too. A run cancelled before its first evaluation keeps `x0`, the honest answer for
"nothing was searched".

Post-fix, same fixture:

| build | polls | evaluations | wall |
|---|---|---|---|
| Windows | **1** | **1** | warm-up dominated (single cold evaluation) |
| WSL | **1** | **1** | **10.4 s** (was 993.3 s) |

The one evaluation is the driver's final context evaluation; the refused poll happens before the
objective runs and costs none.

### 3.4 The test -- `tests/unit/test_audit_optimize.py` [M]

The name is kept so the closure is traceable. The load-bearing bar is now the **evaluation
count**, which no machine, load level or BLAS build can move:

* `polls >= 1` -- the mechanism, and the exact quantity that read 0 pre-fix;
* `evals < evals_ctl`, where `evals_ctl` is an **uncancelled `max_iter=2` run measured in the same
  process** -- comparative, so nothing can rot;
* `evals <= 2` -- one refused poll (costing none) plus the final context evaluation;
* a `DesignResult` still comes back, with `converged is False`;
* `dt < 180 s` is retained **only as a hang-catcher**, explicitly not a performance bar, since
  two evaluations cannot take minutes unless something is stuck.

The full measured history, including the pre-fix 0/404 reading and both builds' wall times, is
recorded verbatim in the test's docstring.

### 3.5 Evidence [M]

| suite | Windows | WSL |
|---|---|---|
| `tests/unit/test_audit_optimize.py` | **89 passed** | **89 passed** |
| the 19 files that touch `design_optimize` / `dual_annealing` | see S6 | -- |

---

## 4. Item 3 -- the idle-box verification bundle

### 4.1 The PMM collateral set, both builds [M]

M3's one open gate: `pytest tests/unit -k "pmm or rcwa or conditioning"`, ~1400 tests. M3 recorded
five abandoned attempts, the best reaching 91 % with zero failures.

| build | result |
|---|---|
| Windows | **1657 passed, 2 skipped, 0 failed** (1:07:45) |
| WSL | S4.1a |

The two Windows skips are both expected and self-describing: `threadpoolctl installed: the cap is
effective here` (the complement of the WSL branch M4 documents) and the `_BLAS_CONTROLLER_LOCK`
exemption row.

**Operational note for whoever repeats this.** Two long runs were lost to argument mangling before
either produced a number: a background `Start-Process -ArgumentList` split
`-k "pmm or rcwa or conditioning"` into separate tokens and pytest collected **nothing** ("no
tests ran in 116.90 s"), and the same class of bug made a WSL launcher exit instantly with empty
logs. A run that reports zero failures because it selected zero tests looks exactly like success
in a tail. Check the collected count, not just the last line.

### 4.2 M4's two solve-heavy suites on WSL [M]

M4's gate was PARTIAL: these two could not complete on WSL under the concurrent-mission load, and
M4 stated the fix is structurally a no-op on that build (no `threadpoolctl`, so `_blas_limit`
returns a null context on both the pre- and post-fix paths) but must still be measured.

```
tests/unit/test_v5_20_8_rcwa_threaded_sweep.py + tests/unit/test_v5_20_2_pmm_jones_2d_jax.py
WSL: 18 passed in 365.40s (0:06:05)
```

18 = 4 + 14, matching M4's Windows counts with leg 2 of the PMM-JAX suite skipped by construction
on a build without `threadpoolctl`. **M4's both-builds gate is now MET.**

---

## 5. Item 4 -- the breadth sweep's new names

WSL chunk 2 produced two names that had not been seen before. Both are **BAD TESTS**, and both are
the same trap as S3's bar.

### 5.1 `test_m1_conditioning_guard.py::test_the_refusal_reproduces_the_prior_answer_with_the_switch`

**It passes in isolation and fails in a sweep**, which pointed at ordering. It is not ordering: it
is the **BLAS thread count**, which the suite pins nowhere. Holding code, build and geometry fixed
and varying only `OPENBLAS_NUM_THREADS` [M]:

**Directly confirmed at the source** [M]: the breadth sweep's own six xdist workers were observed
running at **330-390 % CPU each**, i.e. multi-threaded OpenBLAS, because the sweep command sets no
`OPENBLAS_NUM_THREADS`. So the sweep runs this test at pool > 1 and a bare `pytest <file>` on a
shell that happens to export 1 runs it at pool 1. The two configurations were never the same
experiment.

| threads | `sum(R)` | `sum(R)+sum(T) - 1` | old bars (`>3.0e-2`, `>1.03`) |
|---|---|---|---|
| 1 | 3.216567261e-02 | +3.195613251e-02 | pass -- calibrated here |
| 2 / 4 / 24 / unset | 6.112765047e-03 | +5.903262542e-03 | **both fail** |

**Windows and WSL agree to ten digits at each thread count.** So the widely-quoted framing
"WSL differs" is wrong; this is the near-degenerate draw M1's census exists to record, moving with
the BLAS reduction order. (It also refutes, in passing, the neighbouring X-1 docstring's claim
that 21 TE returns 3.2e-2 "on BOTH builds, agreed to every digit" -- true only at pool 1. That
test asserts only census flags, so it stays green.)

**Fix.** The claim -- with the guard off the call *returns* the wrong pre-M1 answer instead of
raising -- is now stated against a **truncation the census scores clean, solved in the same
process on the same pool**: the clean solves sit at `sum(R)` ~ 2.0e-4 with closure ~1e-13 at every
thread count on both builds. The test asserts the reference really is clean (so it can calibrate),
then that the unguarded answer is `>10x` it in R and non-conserving by `>1e-3` and by `>1e6x` the
clean closure. Measured margins: 30x (pool>1) to 160x (pool 1) against the 10x bar; 5.9e-3 to
3.2e-2 against 1e-3.

### 5.2 `test_niche_audit_w8_raster.py::test_w8_shear_convergence_outlier_vanishes_at_the_coincident_nx`

Reproduces in isolation on WSL: `closure = 7.317e-04` against an absolute gate of `1e-4`. The
gate's own history, in the docstring it shipped with, is the tell -- `<= 2e-07` on the authoring
host, then `2.20e-5` on CI Linux, so the gate was set to `1e-4`. It has now drifted a third time.
The library itself flags this configuration: `RCWAStack.solve` raises `_EnergyWarning`
("the truncation is numerically unstable here") on the 64-slice stack **on both builds**.

Re-tuning the constant again would repeat the mistake, and the measurement says it is unnecessary
-- **the signal is build-independent to five digits while the closure moves 12x** [M]:

| build | closure(64, 128) | `gap` | `neigh` | `gap/neigh` |
|---|---|---|---|---|
| Windows | 6.283e-05 | 1.5516e-04 | 1.5650e-04 | 0.991 |
| WSL | 7.317e-04 | **1.5517e-04** | **1.5650e-04** | 0.992 |

`gap` is the quantity the test exists to measure (pre-fix 1.802e-02, post-fix 1.552e-04 on the
authoring host -- reproduced here to five digits on both builds). The closure error is common-mode
between the two solves being differenced and demonstrably does not contaminate it.

**Fix.** The gate now says what it is actually for: the closure must be small compared with the
**defect this test discriminates** (1.802e-02 pre-fix), so it can neither mimic nor mask it. The
bar is `0.1 x` that defect -- a fixed physical property of the case, not of the machine. The worst
measured closure clears it by 2.5x, and a genuinely broken solve (closure at defect scale) is
still caught.

### 5.3 Chunk collection status

| chunk | result |
|---|---|
| 1 | 3 failed, 1491 passed -- dual_annealing (**fixed**, S3) + the 2 except-budget tests (**owned by D8**, S2.2) |
| 2 | 2 failed, 3030 passed -- both **fixed**, S5.1 / S5.2 |
| 3 | 193 passed, 15 skipped -- clean |
| 4 | 3 failed, 276 passed -- all three are the D8 suite (**owned by D8**; mechanisms in S2.3 / S2.4) |
| 5 | still running at hand-off (4 h 07 m elapsed, `-n 6`) |
| 6 | not started at hand-off |

Chunks 5-6 are the orchestrator's to collect. Nothing in them can be affected by the edits made
here except the two test files named in S5, which are in chunk 2 and are already re-verified.

---

## 6. Green list

| what | Windows | WSL |
|---|---|---|
| PMM collateral set, `-k "pmm or rcwa or conditioning"` (M3's gate) | **1657 passed, 2 skipped, 0 failed** (1:07:45) | **starved, see S7** |
| `test_v5_20_8_rcwa_threaded_sweep` + `test_v5_20_2_pmm_jones_2d_jax` (M4's gate) | 37 passed / 1 skipped (M4) | **18 passed** (6:05) |
| `test_audit_optimize.py` (item 2) | **89 passed** | **89 passed** (6:31) |
| `test_m1_conditioning_guard` + `test_niche_audit_w8_raster`, pool **1** | **49 passed** | **49 passed** |
| the same two files + `test_audit_optimize`, pool **24** | **138 passed** (37:18) | S7 |
| the 19 files touching `design_optimize` (blast radius of S3.3) | **765 passed, 46 skipped, 0 failed** (16:34) | -- |
| `ruff check --no-cache lumenairy/ tests/unit/` | **All checks passed!** | **All checks passed!** |

The pool-1 / pool-24 pair is the load-bearing one for S5.1: pool 24 is the configuration that
produced the original failure, and it is now green on Windows with the two files' 49 tests plus
the optimize suite's 89.

**Known-red, with owner:** the two `test_audit_except_budget.py` tests (S2.2). Red at HEAD, red
now, owned by the D8 session. Nothing was changed to conceal this.

---

## 7. Runs still in flight at hand-off

| run | state at hand-off |
|---|---|
| WSL PMM collateral set (`-n 8 --dist loadfile`) | **1 h 20 m in and starved** -- see below |
| WSL the two S5 files at pool 24 | started, then **deliberately killed** to free BLAS threads for the run above; re-run when the box is idle (it costs ~2 minutes there) |
| WSL breadth-sweep chunks 5-6 | chunk 5 at 5 h 29 m, chunk 6 not started |

**Why the WSL collateral did not finish, measured rather than guessed** [M]. The breadth sweep's
six xdist workers were running at **330-390 % CPU each** (unpinned OpenBLAS, ~21 of 24 cores),
while the collateral's eight workers were getting **11-15 % each**. It is starved, not stuck.

**How much this actually costs the gate is small, and it is worth stating precisely.** The WSL
breadth sweep the orchestrator is already running covers **all of `tests/unit`** on WSL, which is
a strict superset of `-k "pmm or rcwa or conditioning"`. Chunks 1-4 are complete and their only
failures are the eight names accounted for in S5.3. So the WSL breadth coverage of the PMM/RCWA
surface is delivered by chunks 1-4 (plus 5-6 when they land); the dedicated collateral run is a
convenience re-cut of tests the sweep already exercises on that build. What remains genuinely
outstanding on WSL is therefore **chunks 5-6**, not the collateral re-cut.

All five write to
`C:/Users/Tesla/AppData/Local/Temp/claude/C--Users-Tesla/372a2d1f-acbe-4b57-a148-eeae3fe1d729/scratchpad/`
as `gate_collateral_win.log`, `wsl_gates.log`, `verify_wsl_two.log`, `verify_win24.log`,
`optimize_blast_win.log`, `m1_sweep_wsl.txt`.

---

## 8. Uncommitted file inventory

Snapshot **2026-08-05 00:21:15**. The working tree is **LIVE**: a concurrent D8/D9 session is
editing files in it (`test_niche_d9_grid_origin.py` appeared at 23:50, `CHANGELOG.md` at 22:25),
so this is a timestamped snapshot, not a stable fact. Re-take it immediately before tagging.
Modification times are given because they are the only way to tell the campaign's finished work
from the session that is still running.

| file | mtime | attribution |
|---|---|---|
| `lumenairy/elements/pmm/_core.py` | 08-04 19:35 | M2 + M3 |
| `lumenairy/elements/pmm/stack.py` | 08-04 16:42 | M2 + M3 |
| `lumenairy/elements/pmm/stack2d.py` | 08-04 16:43 | M3 (7b.1 BLAS cap) |
| `lumenairy/elements/pmm/conical.py` | 08-04 14:22 | M3 |
| `lumenairy/elements/pmm/_jax_stack.py` | 08-04 15:04 | M3 |
| `lumenairy/elements/rcwa/_core.py` | 08-04 11:39 | M4 |
| `lumenairy/elements/rcwa/stack.py` | 08-04 11:30 | M4 (F-2) |
| `tests/unit/test_v5_20_2_pmm_jones_2d_jax.py` | 08-04 11:20 | M4 (F-1) |
| `tests/unit/test_v5_20_8_rcwa_threaded_sweep.py` | 08-04 14:52 | M4 (F-2) |
| `tests/unit/test_pmm_m2_window_contract.py` (new) | 08-04 12:52 | M2 |
| `tests/unit/test_pmm_m3_efficiency.py` (new) | 08-04 18:46 | M3 |
| `validation/m5_covariant_taper.py` (new) | 08-04 11:10 | M5 |
| `validation/m5_derham_nonuniform.py` (new) | 08-04 09:52 | M5 |
| `validation/m5_taper_degree_spread.py` (new) | 08-04 12:33 | M5 |
| `docs/audits/PMM_M2_WINDOW_CONTRACT_2026_08_04.md` (new) | 08-04 13:05 | M2 |
| `docs/audits/PMM_M3_EFFICIENCY_2026_08_04.md` (new) | 08-04 20:30 | M3 |
| `docs/audits/PMM_M4_HYGIENE_2026_08_04.md` (new) | 08-04 15:46 | M4 |
| `docs/audits/PMM_M5_2D_FEASIBILITY_2026_08_04.md` (new) | 08-04 12:34 | M5 |
| `docs/audits/ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md` | 08-04 11:39 | M4 (D-1) |
| `docs/audits/AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` | 08-04 13:38 | M4 (D-1) + M2 |
| `docs/audits/AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` | 08-04 11:40 | M4 (D-1) |
| **`lumenairy/optimize/driver.py`** | **08-04 22:56** | **this agent (S3.3)** |
| **`tests/unit/test_audit_optimize.py`** | **08-04 23:01** | **this agent (S3.4)** |
| **`tests/unit/test_m1_conditioning_guard.py`** | **08-04 23:31** | **this agent (S5.1)** |
| **`tests/unit/test_niche_audit_w8_raster.py`** | **08-04 23:33** | **this agent (S5.2)** |
| **`docs/audits/PMM_GATE_CLOSURE_2026_08_04.md`** (new) | **08-05 00:19** | **this agent** |
| `lumenairy/propagators/carrier.py` | 08-04 22:22 | **NOT this campaign** -- live D8/D9 session |
| `lumenairy/elements/_lens_traced.py` | 08-04 22:02 | **NOT this campaign** -- live D9 session |
| `tests/unit/test_niche_d6_exact_tilted_leg.py` | 08-04 21:56 | **NOT this campaign** -- live D9 session |
| `tests/unit/test_niche_d9_grid_origin.py` (new) | 08-04 23:50 | **NOT this campaign** -- live D9 session |
| `CHANGELOG.md` | 08-04 22:25 | **NOT this agent** (instructed not to touch it) |

**Five files in this tree belong to a mission that is still running** (the last four rows plus
`CHANGELOG.md`). The 5.33.0 train should either wait for that session to declare done or tag
without them; they are not covered by any gate in this document, and `carrier.py`'s D8 surface is
the one carrying the known-red budget tests (S2.2).

**Files this agent changed, and nothing else:**

| file | change |
|---|---|
| `lumenairy/optimize/driver.py` | S3.3 -- `_DualAnnealingCancelled` + the objective-level cancellation poll |
| `tests/unit/test_audit_optimize.py` | S3.4 -- evaluation-count bar replacing the wall-clock bar |
| `tests/unit/test_m1_conditioning_guard.py` | S5.1 -- in-process clean-truncation reference replacing two thread-dependent absolute bars |
| `tests/unit/test_niche_audit_w8_raster.py` | S5.2 -- comparative closure gate |
| `docs/audits/PMM_GATE_CLOSURE_2026_08_04.md` | this document |

`lumenairy/propagators/carrier.py` was edited and then **reverted byte-exact** (S2.1); it is not
in this agent's inventory. `CHANGELOG.md` was not touched, per instruction.

---

## 9. Lessons worth carrying

1. **A green suite is not evidence when the bar is a machine property.** Four of the five failures
   this session were absolute bars on BLAS-thread-count- or build-dependent magnitudes. The
   durable form is always the same: measure the reference *in the same process, on the same pool*,
   and assert a ratio.
2. **An absolute bar can hide a total feature failure.** dual_annealing's cancellation did
   literally nothing -- 0 polls, the full 404-evaluation budget -- and a 30 s wall-clock assertion
   called that a pass for five releases, on the faster of the two builds.
3. **The evaluation count is the load-independent twin of a wall-clock bar.** The same 404
   evaluations took 47.6 s and 993.3 s on the two builds. Counts do not have a spread.
4. **Validate the instrument before believing a null reading.** "0 polls" and "the counter is not
   wired" are indistinguishable until a positive control says otherwise.
5. **Never revert another session's file wholesale.** The retraction's suggested
   `git checkout -- <file>` would have destroyed 257 lines of live, uncommitted work in that same
   file. Diff first; revert hunks.
6. **A test that passes alone and fails in a sweep is a fact about shared state, not about
   flakiness** -- here it was the BLAS pool in one case and an unpicklable module-level glass
   registration in another, and both turned out to name a real defect.

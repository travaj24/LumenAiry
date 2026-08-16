# Test-hygiene chore -- 2026-08-16

Branch `chore/test-hygiene`.  Three scoped items, no library code touched.
Binding law: `docs/TESTING_STANDARDS.md`.

| item | verdict |
|---|---|
| 1 -- `.test_durations` surgical regen | **DONE** -- 217 measured entries added, 263 stale deleted, 11,931 retained byte-for-byte |
| 2 -- the OMP-preamble defect | **REFUTED as stated** -- 0/61 files have the in-file ordering defect; the guards are inert for a different, unfixable-in-file reason, already documented in `tests/conftest.py` |
| 3 -- OOP fast-path validation gap | **DONE** -- 7 tests added to `tests/unit/test_p2c_pmm2d_stack_cascade.py` (section 8) |

Everything below was measured on this box (Windows 11, 24 cores, py3.14.6 /
numpy 2.4.4) and, where stated, on WSL (py3.12.3 / numpy 2.5.1 and the
numpy 1.26.4 "floor" venv, openblas64 0.3.23).

---

## Item 1 -- `.test_durations` surgical regeneration

### (a)-(c) what the file was carrying

`.test_durations` held 12,194 entries.  The two gates that consume it are

```
pytest tests/unit -m "not integration and not slow"   # fast gate, --splits 5 (publish.yml) / 3 (unit-tests.yml)
pytest tests/unit -m "slow and not integration"       # slow gate, --splits 3
```

which together collect 12,148 ids on this tree (11,914 fast + 234 slow).
Diffing the two sets:

* **217 MISSING** (collected, no duration)
* **263 STALE** (duration, no longer collected)

| MISSING, by file | n |
|---|---|
| `test_obl_banded_halo.py` | 87 |
| `test_tangent_facet.py` | 68 |
| `test_p2c_pmm2d_stack_cascade.py` | 34 (27 shipped + 7 added by item 3) |
| `test_fix_runner_oom_2026_08_13.py` | 14 |
| `test_eme_census_determinacy.py` | 3 (the restructured ids) |
| `test_niche_audit_w9_eig_vjp.py` | 2 |
| `test_niche_audit_e_prepared_and_enums.py` | 2 |
| `test_niche_r0_byte_budgeted_cache.py` | 2 |
| `test_audit_propagation.py`, `test_niche_k3_perf.py`, `test_niche_r4_fga_dual_vectorize.py`, `test_pmm_m2_window_contract.py`, `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` | 1 each |

The 263 stale ids are dominated by `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`
(214, a parametrization that shrank) and are worth 72.9 s in total.

### (d) measurement

One quiet single-threaded run **per file**, 13 files, on an otherwise idle
box, with `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1` exported **before**
the interpreter started, using pytest-split's own store:

```
python -m pytest <file> -m "not integration" -q -p no:cacheprovider \
    --store-durations --clean-durations --durations-path <scratch>/<file>.json
```

All 13 exited 0.  `--clean-durations` + a per-file path means each run stores
only its own tests, so nothing else in the committed file could be perturbed.

### (e) splice

Line-based, not a JSON round-trip: every retained entry's `"key": value` text
is copied verbatim from the committed file, so the only lines that differ are
the 217 added and the 263 removed (plus the comma on the final line).  Checked
programmatically: **11,931 retained entries, 0 byte-changed**.  The result is
still `sort_keys=True, indent=4` -- exactly what pytest-split's own writer
emits -- and still CRLF in the working tree / LF in the blob, matching
`core.autocrlf=true`.

### (f) validation -- the 5-shard split, before and after

Two facts about pytest-split matter here:

* `_remove_irrelevant_durations` drops entries for ids that are not collected
  **before** splitting.  So the 263 stale entries never mis-weighted a shard
  directly; their only effect was on the fallback average below.
* `_get_items_with_durations` prices a **missing** id at the *average* of the
  relevant entries -- 0.5843 s here.  That is where the damage was.

`LeastDurationAlgorithm` run over the current collected item list, with each
candidate durations file as the weights, priced against the measured file:

| gate | splits | file | believed max/min | REAL max/min |
|---|---|---|---|---|
| fast (release verify) | 5 | committed | 1.000 | **1.048** |
| fast (release verify) | 5 | spliced | 1.000 | **1.000** |
| fast (unit-tests.yml) | 3 | committed | 1.000 | 1.016 |
| fast (unit-tests.yml) | 3 | spliced | 1.000 | 1.000 |
| slow | 3 | committed | 1.000 | 1.012 |
| slow | 3 | spliced | 1.000 | 1.000 |

The aggregate ratio understates the defect, because the 216 fast-gate
misses (126.2 s of fallback against 181.0 s of real work, +54.8 s
unattributed) contained a handful of very large individual errors that the
balancer had no way to see:

| test | fallback | measured |
|---|---|---|
| `test_eme_census_determinacy.py::test_the_recovered_mode_is_confirmed_by_the_fd_oracle_not_by_the_prefix` | 0.58 s | **36.61 s** |
| `..._determinacy.py::test_a_straying_polish_cannot_lose_a_mode_the_minimiser_already_had` | 0.58 s | **34.84 s** |
| `test_fix_runner_oom_2026_08_13.py::test_the_release_blocker_call_fits_a_runner_sized_box` | 0.58 s | **33.09 s** |
| `..._determinacy.py::test_the_fixed_census_is_nudge_invariant_and_a_tie_at_the_cut_flips_the_prefix` | 0.58 s | **28.61 s** |
| `test_niche_audit_w9_eig_vjp.py::test_the_theta0_defect_pin_fires_when_the_defect_is_absent` | 0.58 s | 8.09 s |

i.e. four tests priced at 60x under their real cost, all of which the
least-duration heap would happily stack onto one shard.  Two files moved the
other way (`test_obl_banded_halo.py` -44.6 s, `test_tangent_facet.py` -33.4 s:
87 and 68 tests that are individually far *cheaper* than the average), so the
sums partly cancelled -- which is exactly why the aggregate max/min looked
benign while a single shard could still be handed three 30-second tests.

End-to-end check with the real plugin, `--splits 5 --group N`:
2381 + 2383 + 2383 + 2383 + 2384 = **11,914**, an exact partition of the fast
gate.

### Still open (deliberately out of scope)

The *retained* entries remain on the scale they were captured at.
`AUDIT_CI_TEST_TIME_2026_08_03` S2 and the comments in both workflows record
that the fast-gate half of this file was captured with BLAS unpinned on a
many-core desktop, which inflates eig-bound tests by 1-2 orders of magnitude
relative to subprocess-bound ones.  A surgical splice cannot fix that; only a
full serial single-threaded regeneration of both gates can, and that was
explicitly not this item.  The 217 new entries **are** on the pinned scale, so
the file is now *less* internally consistent in one narrow sense and much more
accurate in another -- worth stating plainly rather than claiming the file is
now clean.

---

## Item 2 -- the OMP-preamble defect: REFUTED as stated, confirmed in effect

### The claim

Some `tests/unit` modules call `os.environ.setdefault("OMP_NUM_THREADS", "1")`
*after* numpy is imported in module order, so the cap binds nothing.

### The sweep

AST scan of all of `tests/` (`grep --include='*.py'` seeded, then parsed):
**65 files** mention a thread env var; **61** contain a module-level statement
that *sets* one (the other 4 mention it in prose or build a subprocess `env`
dict: `tests/conftest.py`, `test_niche_audit_w3_rcwa_pmm.py`,
`test_niche_r3_gbd_mem_lstsq.py`, `test_v5_24_3_jax_ci_coverage.py`).

For each of the 61, the scan walked the module body in order and compared the
line of the env statement against the line of the first import that
transitively loads numpy (transitivity resolved empirically: each imported
module was imported in a fresh interpreter and `'numpy' in sys.modules`
checked).

> **Result: 0 of 61 files have the defect.**  In every one, the env block is
> above every numpy-loading import.  Two files
> (`test_audit_p1_gui_dead_import.py`, `test_audit_w6_ui.py`) import no
> numpy-loading module at all.

**No file was modified for this item**, because there is nothing to move.

### The probe (per file, both arms)

Because "the scan found nothing" is not evidence, each of the 61 files was
probed in two fresh subprocesses with `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB`
*unset* in the parent environment:

* **asis** -- import the file exactly as committed;
* **moved** -- import a temp copy with the env block *relocated* to just after
  the last module-level import (the defect shape, constructed on purpose --
  the fail-before demonstration that the probe is a detector and not a
  tautology);

then read `threadpoolctl.threadpool_info()`.

| (asis, moved) max num_threads | files |
|---|---|
| (1, 24) | 44 |
| (2, 24) | 10 |
| (4, 24) | 5 |
| (4, 4) | 2 |

59/61 discriminate: as committed the preamble binds the count the file asks
for (1, 2 or 4); relocated below the imports it binds nothing and OpenBLAS
runs at the box's 24 cores.  The two (4, 4) files are the two that import no
numpy-loading module, so there is no ordering for the probe to break -- they
ask for 4 and get 4 in both arms.  (Three files raise `ImportError` on a
direct import because they use relative imports; their env block executes
before that point, so both arms still read out cleanly.)

### What IS true, and why no in-file edit can fix it

The group-D observation that the guards cap nothing *under pytest* is correct.
The cause is not in-file ordering -- it is that pytest imports
`tests/conftest.py` first, and that file does `import numpy as np` /
`import lumenairy as la` at module scope, so OpenBLAS has already latched its
thread count before any test module's body runs.  Three arms, same file,
`OMP_*` unset in the parent:

| arm | numpy already imported | env `OMP_NUM_THREADS` after import | OpenBLAS `num_threads` |
|---|---|---|---|
| A: `pytest tests/unit/test_farfield.py --collect-only` | True | `'1'` (the guard DID run) | **24** |
| B: same, `OMP_NUM_THREADS=1` exported before python | True | `'1'` | **1** |
| C: same module imported directly, no conftest | -- | `'1'` | **1** |

Arm A is the defect in effect: the guard sets the variable and OpenBLAS
ignores it.  Arm C is the proof the *file* is not what is wrong.  Arm B is the
fix that works, and it is the fix the repository already uses -- the workflow
`env:` block, which `tests/conftest.py` lines 24-53 document at length
("A NO-OP GUARD IS NOT A PIN ... The pin is therefore made where it actually
takes effect ... To pin locally, set the variables in the shell that launches
pytest"), and which `test_v5_24_3_jax_ci_coverage.py::test_s4_4_jax_job_pins_blas_threads_to_avoid_openblas_deadlock`
already pins for the jax lane.  `unit-tests.yml`'s `slow-tests` job pins it
too.

So the ~50x-slow local/WSL runs the group-D note measured are real, and the
remedy is operator-side (export the caps before `python`), not a source edit.
Nothing was changed, and no structural test was added: a pin on "the env line
precedes the numpy import" would only be load-bearing for running a test file
as a script, which nothing does.

---

## Item 3 -- out-of-plane tensor layers in the cascade oracle matrix

### The gap

`BUILD_PMM2D_CASCADE_2026_08_16.md` S4 records a 360-case campaign matrix that
*did* include an `out-of-plane tensor` axis -- but that was a one-off script.
The **shipped** `tests/unit/test_p2c_pmm2d_stack_cascade.py` could not even
construct such a layer: its `_stack` helper routed everything to `eps` or
`eps_cell`, so all 27 tests ran on `('sym', W, V, lam)` modal sets, i.e. the
Redheffer branch of `solve()`.

That left the *entire second copy* of the P2C machinery uncovered in CI.  A
layer with `eps_xz`/`eps_yz` != 0 breaks the `[W; -V] <-> -lam` symmetry and
promotes the whole stack to the generalized branch, which has its own dedup
and its own merge: `_Mmemo`, `_gifc`, `_cascade_sequence_general`,
`_interface_smatrix_general`, `_propagation_smatrix_general`.

### What was added (section 8, 7 tests)

Device: a tilted-axis uniaxial pillar in an isotropic host,
`eps = eps_o I + (eps_e - eps_o) u u^T` with
`u = (sin psi cos chi, sin psi sin chi, cos psi)` -- a real crystal
orientation rather than an arbitrary matrix, so `eps_xz` and `eps_yz` are
both nonzero by construction and `eps_zz` (which the library requires) is
automatically nonzero.  Two distinct orientations `_OOP_A`, `_OOP_B`.

The identity contracts mirror section 1 arm for arm, as required:

| test | contract | why that contract |
|---|---|---|
| `test_p2c_an_oop_tensor_layer_promotes_the_whole_cascade` | decisions | `eps_xz/eps_yz != 0`; modal set is `'gen'`; a mixed `[scalar, oop]` stack reads `['sym','gen']`; and `solve(retain_internal=True)` **raises** -- the raise is the observable readout that the generalized branch was selected, rather than reading a private flag |
| `test_p2c_oop_fast_matches_monolithic_across_incidence[3]` | derived bar | mergeable stack; normal / oblique / conical |
| `test_p2c_oop_repeated_layer_dedup_is_bit_for_bit_identical` | **exact equality** | A-B-A-B: repeats are not adjacent, so nothing merges and only the dedup is in play -- same bytes, same LAPACK.  Mirrors `test_p2c_distinct_layer_stacks_are_bit_for_bit_identical` |
| `test_p2c_oop_merged_run_equals_one_explicitly_thick_layer` | derived bar | independent oracle: one layer of thickness `t1+t2`.  Mirrors section 1's merge oracle |
| `test_p2c_oop_merge_bar_has_a_gap_on_both_sides` | rule 5 | the new bar's two-sided gap |

The **repeated-OOP-layer dedup** case carries three claims, all
build-free: 4 layers / 2 distinct modal keys, `_build_layer_modes` entered
exactly **twice** (counted by wrapping the bound method -- a decision, not a
wall-clock reading); the generalized cascade sequence stays 4 entries long
(nothing merges); and the answer is bit-for-bit the monolithic answer at
normal and conical incidence.  The adjacent-repeat case asserts the
complementary decision: `_cascade_sequence_general(..., merge=True)` collapses
`T-T` to 1 entry while `merge=False` keeps 2.

### The bar

Re-derived for this branch, because the quantity that bounds the merge here is
`_interface_smatrix_general(M, M)` (which is `(0, I, I, 0)` in exact
arithmetic), not `_interface_smatrix(W, V, W, V)`.  Same shape as
`_merge_bar`: `1e3 * (2n+2) * residual`.

Measured on this build, `n_orders=4`, `degree=7`:

| quantity | measured |
|---|---|
| generalized identity residual, normal / oblique / conical | 3.500e-15 / 3.910e-15 / 2.880e-15 |
| fast vs monolithic, 5-layer mergeable OOP stack | 7.772e-16 / 1.221e-15 / 9.806e-15 |
| derived bar for that stack | ~3.5e-11 .. 4.7e-11 |
| gap check: noise / bar / smallest real signal | 3.331e-16 / 2.800e-11 / 9.149e-04 |
| merged OOP run vs one explicitly thick layer | 0.000e+00 |

Both gaps clear two decades with room (noise->bar ~4.9 decades,
bar->signal ~4.5 decades).

### One assertion re-derived mid-build (worth recording)

The first draft of the promotion test asserted per-polarization
`R+T <= 1 + 1e-9`, copied from the lossy-overflow test.  It **failed**: the
mixed stack reads `R+T-1 = 2.29e-4`.  Re-measuring rather than loosening
found the cause -- the file's own 6x6 / degree-7 scalar cell carries that
defect by itself:

| device | n_orders 2 / 4 / 6 / 8, normal + conical | `max abs(R+T-1)` |
|---|---|---|
| scalar-only (the existing `_cell(12.0, 2.1)`) | | 5.7e-5 .. 2.5e-4 |
| out-of-plane-only | | 1.5e-6 .. 6.0e-5 |
| mixed | | 1.3e-4 .. 5.4e-4 |

i.e. `1e-9` was an assumed number pinning the *discretization*, and the
out-of-plane layer is an order of magnitude *better* behaved than the scalar
one this file has always used.  Restated build-free as a **ratio** against the
same device's scalar baseline measured inside the test: the promotion must not
degrade energy balance beyond what the scalar branch already loses.  Measured
ratio 1.9-2.3; bar 100x (~1.6 decades above the observed ratio, ~1.7 below the
O(1) defect a broken generalized cascade would give).  This arm is also the
one thing a fast-vs-monolithic comparison structurally cannot do -- both
branches would be wrong together.

### Runs

`tests/unit/test_p2c_pmm2d_stack_cascade.py`, 34 tests (27 + 7), green on
every arm:

| mount | interpreter / numpy | BLAS threads | result |
|---|---|---|---|
| Windows `C:\tmp\lum_hy` | py3.14.6 / numpy 2.4.4 | 1 | 34 passed (18.2 s) |
| Windows `C:\tmp\lum_hy` | py3.14.6 / numpy 2.4.4 | 2 | 34 passed (16.4 s) |
| WSL `/mnt/c/tmp/lum_hy` | py3.12.3 / numpy 2.5.1 | 1 | 34 passed (21.3 s) |
| WSL `/mnt/c/tmp/lum_hy` | py3.12.3 / numpy 2.5.1 | 2 | 34 passed (14.8 s) |
| WSL floor venv | py3.12.3 / numpy 1.26.4, openblas64 0.3.23 | 1 | 34 passed (18.5 s) |

Three distinct LAPACK builds; the bit-for-bit arms hold on all of them.
`ruff check` clean.

# CI PREMISE GATES -- 2026-09-11

Final repair round for the 5.45.0 release matrix (`main` `3865813`, matrix run
`34566427386`, **RED on twelve tests**, not tagged).  Working tree
`fix/ci-premise-gates` off `3865813`.

This round changes **no guard rule and no library arithmetic**.  Every RULE the
five guards apply is correct and ships unchanged; the one library edit is a
docstring correction in `lumenairy/elements/pmm/_core.py` (item B below).  What
changes is what the *tests* assert.

---

## 1. The fact that drives this round

The twelve failures are not twelve bugs.  They are one fact, seen twelve times:

> **On the CI runner arm, the ill-conditioned solves that read WRONG on every
> local kernel, thread width and build come out CORRECT.**

The guards then, correctly, take different decisions there -- they return
instead of refusing, stay silent instead of warning -- and every assertion that
was written as "this fixture reproduces the pathology" failed.  A guard whose
job is to refuse wrong answers and return correct ones **must follow the
answer**; a test that demands a fixed outcome demands that it stop.

The evidence, read out of the downloaded artifacts (`C:/tmp/ci_5450b/`, one
`unit_test_output.txt` per python x shard plus `jax-unit-output/`):

| what CI measured | CI reading | every local arm |
|---|---|---|
| plain-1-D interface, wall separation 1e-04, guard disarmed | `R+T` = 1.0000003658, 0 warnings | 0.9999999898 .. 1.0000000000 (correct too) |
| plain-1-D interface, wall separation **1e-05**, guard disarmed | `R+T` = **1.0000010472** (correct) | **1.17 .. 3.62** (wrong) |
| 1-D sliver guard at 1e-05 | **returns**, silent | **refuses** |
| LC / OOP-director / gyrotropic sliver at 3e-05 | all three `right`, all three `silent` | at least one `wrong` |
| 5-layer per-layer window staircase | err/delta 4.15 / 4.39 / 4.47, all `silent`, `R+T` 1.0000000002 / 1.0000009842 / 1.0000052617 | at least one arbitrated |
| M1 X-1 engineered pre-round-1 arm | **0.267x** (py3.10 sh3, py3.11 sh1) / **0.677x** (py3.10 sh3, JAX lane) of the converged 2.094088e-04 | **152.6x** |
| branch-cut spacer-detune pre-round-1 arm | worst spread **5.551e-15** against the repaired 3.331e-15 -- FLAT | 1e3x+ larger |
| owned-liner ladder at 1e-07, 4 degrees | `R+T` 1.2652 / 1.2058 / 1.2684 / 1.1210, spread **1.1315** | spread > 1.2 |
| guarded mortar solve vs `np.linalg.solve` | py3.13 wheel ONLY: `b2eb080e...` vs `5495e552...` | identical |
| `test_ci_kernel_consistency` | arm `WSL-unknown-t1` disagrees on 3 rows -- `answer@1e-05` closes/open, `warned@1e-05` silent/warn, `sliver/pmm1d@1e-05` return/refuse | -- |

Those three census rows are exactly the rows where **CI is right and we are
wrong**.  The census was comparing OUTCOMES where it had to compare RULES.

---

## 2. What is known, and what is not, about the CI arm -- OPEN

**This round does not solve it and does not try to.**  It is recorded here as
an open item so the gates' skip reasons point somewhere.

### 2.1 What the artifacts actually say

Extracted from every `unit_test_output.txt` and `jax_unit_output.txt`:

* `platform linux -- Python 3.10.21 / 3.11.16 / 3.12.14 / 3.13.15,
  pytest-9.1.1, pluggy-1.6.0`, interpreters from
  `/opt/hostedtoolcache/Python/<v>/x64/bin/python`;
* `rootdir: /home/runner/work/LumenAiry/LumenAiry`, `configfile: pyproject.toml`;
* plugins `split-0.11.0, benchmark-5.3.0, cov-7.1.0, xdist-3.8.0, zarr-3.3.0`
  (the JAX lane adds `timeout-2.4.0`, `timeout: 200.0s`);
* **`threadpoolctl` is NOT installed on the runner.**  Stated by the runner
  itself, in a warning raised from `lumenairy/elements/rcwa/_core.py:224`:
  "`rcwa: set_blas_threads(...) ... needs the optional threadpoolctl package,
  which is not installed -- the requested BLAS-thread cap is INERT`".  This is
  why the census arm identifies as `WSL-unknown-t1`: the probe reads the kernel
  back out of the loaded library and there is nothing to read it from.

### 2.2 What the artifacts do NOT say

* **No numpy or scipy version is printed anywhere in these artifacts.**  There
  is no "Library import sanity" step in the downloaded logs (they are pytest
  output only), and `numpy.show_config` is never called.  The workflow installs
  unpinned (`pip install -e ".[fft,perf,numba,hdf5,zarr,dev]"`), so the wheel
  set is whatever PyPI served on the day.  The round's premise records numpy
  2.4.6 / scipy 1.17.1 per python 3.10-3.13; **that is not verifiable from the
  logs in hand** and is recorded as such in the synthetic census arm.
* No CPU model is printed.  AMD EPYC 7763 (Zen 3) is the ubuntu-latest runner
  fleet's usual part and is the round's premise; the logs do not confirm it.
* No effective BLAS thread width is recorded.  The fast lane deliberately does
  not pin BLAS in the environment, and the test modules' own
  `os.environ.setdefault("OMP_NUM_THREADS", "1")` runs at *module import*, which
  is only effective if OpenBLAS has not already been loaded by an earlier test
  module in the same process.  Whether it had been is not recoverable from the
  logs.

### 2.3 What has been ruled OUT here, by measurement

| hypothesis | measurement | verdict |
|---|---|---|
| a Zen-specific BLAS micro-kernel | `OPENBLAS_CORETYPE=ZEN` is not in the DYNAMIC_ARCH table in these wheels; it and an unrecognised name both fall back to auto-detection, which reports `Haswell` on this Zen 3 host.  The local default arm IS the Zen arm at kernel level | not the cause |
| the BLAS micro-kernel at all | four kernels measured -- `Haswell`, `Sandybridge`, `Nehalem`, `Katmai` (`KATMAI` and `PRESCOTT` both resolve to `Katmai`) -- on two builds, 10 arms.  All ten read the 1e-05 row **wrong** | not sufficient |
| the thread width | `t1` and `t4` arms on both builds; all read it **wrong** | not sufficient |
| the numpy version | WSL runs **numpy 2.4.6**, the version the round records for CI, and reads it **wrong** | not sufficient |
| the OS / libm | WSL is Ubuntu on the same silicon and reads it **wrong** | not sufficient |
| AVX-512 | `SKYLAKEX` is unreachable on this Zen 3 host (SIGILL on the first BLAS call, exit 132) and EPYC 7763 has no AVX-512 either, so neither side is running it | not available as an explanation |

What remains unexcluded: the **wheel set** (numpy and scipy builds, and their
bundled OpenBLAS versions, as served to each CI python), the interpreter
version, and the possibility that the runner's process had OpenBLAS already
loaded at a different thread width.  **OPEN.**

### 2.4 The one mechanism this round did pin down

`test_the_guarded_mortar_solve_returns_the_numpy_solve_bit_for_bit` failed on
**py3.13 only**, on a WELL-conditioned operand (cond = 3.3e+02), with two
different byte sequences for the same solve.  That one has an explanation, and
it is measurable here:

> **NumPy's `np.linalg.solve` and SciPy's `lu_factor`/`lu_solve` go through
> DIFFERENT bundled OpenBLAS/LAPACK builds.**

Measured 2026-09-11 on both local builds, via `show_config(mode="dicts")` and
confirmed by `threadpoolctl.threadpool_info()` listing **two** distinct loaded
libraries in one process:

| package | bundled LAPACK | notes |
|---|---|---|
| numpy (WIN 2.4.4 / WSL 2.4.6) | `scipy-openblas 0.3.31.188.0`, `USE64BITINT DYNAMIC_ARCH`, build target `Haswell` (WIN) / `SkylakeX` (WSL) | `libscipy_openblas64_-*.dll` / `.so` |
| scipy 1.17.1 | `scipy-openblas 0.3.30`, `DYNAMIC_ARCH`, build target `Haswell` | `libscipy_openblas-*.dll` / `.so` |

(The build target in that string is the build's *maximum*, not the runtime
dispatch: both libraries dispatch `Haswell` on this host, read back from
`threadpoolctl`, and both respond to `OPENBLAS_CORETYPE`.)

So **NumPy-vs-SciPy bit-identity is not a portable claim**, and it never was --
it survived the local gate only because two different OpenBLAS versions happen
to agree bit for bit on these operands here.  The library's docstrings said it
was measured; they now say where it holds.

---

## 3. The restatement contract

Every test restated in this round is split in two:

* **INVARIANT assertions -- unconditional, on every arm.**  The repaired or
  guarded path returns an answer that agrees with the sliver-free / analytic /
  converged reference within the derived bar; a row the guard refused was
  measured wrong; a row it warned about is inside the band *by geometry*; the
  screen reaches the path at all; the switch moves no bit.
* **PREMISE-GATED assertions.**  Anything whose premise is a *numerical reading
  of a pathology* -- "the pre-fix arm reproduces X", "this fixture is wrong on
  this build", "the hazard band contains this row", "there is a super-unity
  here", "bit-identical to numpy".  The premise is **measured on the running
  arm first**, and if it does not hold the test `pytest.skip`s with a reason
  string that **carries the reading**.

Three rules, and they are what keeps this from being a way to make red go away:

1. **Never `assert` the premise.**  A premise that is asserted is a test of the
   runner, not of the library.
2. **Never relax a bar and never delete an arm.**  Every bar in this round is
   unchanged; the fail-before arms are all still built and still run.
3. **A skip is visible; a false assertion is a red shard.**  The matrix summary
   shows `SKIPPED [n] file:line: premise absent on this arm: <reading>`, which
   is a measurement in the log, not a silence.

### Relation to `docs/TESTING_STANDARDS.md`

Rule 4 there says *"Never `pytest.skip` on a resource check -- two skips
silently removed five tests from the gate on exactly the runners that
mattered."*  That rule is about an **environment precondition** (a worker pool,
a big box) standing in for the claim itself, which removes the *whole* test.
Here the claim is split first: the library-facing half stays unconditional on
every arm, and only the fail-before demonstration is gated -- and it is gated on
a **measurement of the physics**, printed in the skip reason, not on a property
of the machine.  Rule 3 ("engineer the state; don't hope the build produces it")
is still the preferred shape and is still used where the state *can* be
engineered: what this round adds is what to do when the engineered state is
faithfully constructed and the arm's arithmetic **declines to break**.

---

## 4. The twelve tests, and the shape each became

| # | test | new shape |
|---|---|---|
| 1 | `test_m1_conditioning_guard.py::test_the_withdrawn_refusal_moves_no_bit_and_the_ladder_carries_no_silent_defect` | (a) shipped ladder carries no silent defect and (b) the switch is a no-op on the historical cell stay UNCONDITIONAL.  (c) the engineered pre-round-1 arm: the switch's bit-identity on the broken row is asserted whenever the arm produces one; the **152x magnitude** is gated at 10x and skips with `M`, pol, `sum(R)` and closure.  A new paired assertion behind the gate: on an arm that does reproduce it, the SHIPPED body closes **that same cell**. |
| 2 | `test_verify_pmmstack_sliver_walls.py::test_the_remedy_lands_on_the_structures_own_continuity_slope` | the prescribed `min_feature` is now DERIVED from the geometry (`_prescribed_mf`, the same `2 * w_wide * period` the refusal text builds) instead of parsed out of a refusal, so the slope bar and the closure of the snapped answer are asserted on every arm.  Gated: (1) that the guard refuses the un-snapped row at all -- and, behind that gate, that the `min_feature` it NAMES equals the derived one; (2) that a fixed `2 delta` bar fails on one of the two fixtures. |
| 3 | `test_verify_pmmstack_sliver_walls.py::test_a_thin_feature_owned_by_one_layer_is_exempt_and_that_is_not_free` | the ownership rule's silence (pure geometry), the never-refuses contract and continuity at 1e-03/1e-05 stay unconditional.  `max(err) > 0.1` -- a bar CI cleared by 1.1x -- is replaced by the scale-free **`min(err) > 100 * delta`** at 1e-07 (four decades of gap, worst CI reading 1.09e+06 x delta).  The degree-to-degree `R+T` spread `> 1.2` is gated and skips with all four readings; the super-unity claim sits behind that gate. |
| 4 | `test_verify_pmmstack_sliver_round3.py::test_round4_closes_the_restorable_wrong_answer_r3a_left_returned` | the D-5-band premise (`_SLIVER_ATTRIB_CLOSURE < floor < _SLIVER_TRIGGER_BAR`) is measured rather than asserted and joins the wrong-as-returned / right-when-snapped premise; `assert seen` becomes a skip listing, per mount, the floor, whether it is in band, `err/delta` returned and snapped, and the round-3 drop. |
| 5 | `test_verify_pmmstack_sliver_round3.py::test_a_correct_answer_is_refused_when_the_snap_leaves_the_superunity_regime` | every row that meets the premise is still fully arbitrated and asserted; `assert refused >= 2` becomes a skip carrying `(delta, pol-1 class, err/delta, R+T)` for all three directed wall steps. |
| 6 | `test_verify_pmmstack_sliver_round2.py::test_the_within_layer_arm_is_silent_where_the_theorem_lets_it_be` | the ownership rule and "never refused" are asserted on all eight (width, degree) pairs.  Both population readings -- the arm speaks somewhere, the arm is silent on a ruined answer somewhere -- are gated and skip with the full 8-row table. |
| 7 | `test_verify_branch_cut_round2.py::test_the_repaired_answer_does_not_depend_on_a_spacer_detune` | POST (the repaired closure is detune-independent under `_DETUNE_FLATNESS_BAR`) is unconditional.  The PRE contrast is gated and skips with both spreads per truncation. |
| 8 | `test_fix_pmmstack_sliver_walls_round2.py::test_the_per_layer_window_path_is_arbitrated_on_its_OWN_grid` | the pairing (wrong is never silent, right is never refused) and "the screen reaches this path" are unconditional; "the guard says something here" is gated and skips with `(delta, err/delta, outcome, R+T)`. |
| 9 | `test_fix_pmmstack_sliver_walls_round2.py::test_the_guard_now_reaches_a_liquid_crystal_sliver` | provable passivity of all three tensor classes, the screen's reach, the pairing, the sliver-free control and the non-Hermitian payload's never-refused contract are unconditional (the last was moved ahead of the gate).  "At least one class is WRONG" is gated and skips with the three `(class, continuity class, outcome)` triples; the pol-0 / pol-1 disagreement sits behind the same gate. |
| 10 | `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` | unconditional: the site is structurally unguarded (two bare `np.linalg.solve`, no screen), the correct row closes under 1e-04 and is silent, `rcond` is already under 1e-09 there, the two populations are under **2.1 decades** apart, the committed census's 1e-12 verdict is non-unanimous, and this arm's own verdict is one of the two.  Gated: that the 1e-05 row is WRONG at all; skips with `R+T`, both `rcond`s and the decade gap. |
| 11 | `test_fix_pmm2d_mortar_round2.py::test_the_guarded_mortar_solve_returns_the_numpy_solve_bit_for_bit` | restated to what is portable, on all six (delta, M) rows: (1) the guarded answer is **bit-for-bit its own unguarded `lu_factor` + `lu_solve`** -- the screen adds nothing; (2) relative residual `<= 16 * n * eps` (measured 0.017 .. 0.050 of `n*eps` over 4 kernels, 324x of gap); (3) agreement with `np.linalg.solve` within `64 * cond(A) * eps * max|x|` (measured 0 against a bound reaching 2.4e-07).  The numpy bit-identity is kept as a PREMISE-GATED informative check that skips when numpy and scipy do not resolve to one LAPACK build -- detected by `show_config(mode="dicts")` with `threadpoolctl` as corroboration.  **It skips on every arm measured, including both local builds**, which is the finding. |
| 12 | `test_ci_kernel_consistency.py::test_this_arm_agrees_with_the_committed_census` | the census contract moves from OUTCOME equality to RULE conformance -- see section 5. |

### 4.1 Other tests in the same six families, same shape

Found by grepping the seven sliver files, the M1 file, the branch-cut and
rcwa-even-sector fix/verify files, the mortar round-2/3/4 files and the
consistency test for assertions whose premise is a numerical reading of a
pathology:

| test | gated claim |
|---|---|
| `test_fix_branch_cut_round2.py::test_the_coincident_spacer_ladder_is_passive_and_the_pre_arm_is_not` | the pre-round-1 arm breaks a spacer mount by 1000x the control (the same shape as #7).  The NO-SPACER control's cleanliness stays unconditional. |
| `test_fix_branch_cut_round2.py` (PMM on-cut census) | `total > 0` -- the fixture must actually put modes on the cut.  `bad == 0` (no on-cut mode carries the incoming root) stays unconditional. |
| `test_fix_pmmstack_sliver_walls.py::test_the_guards_DECISION_is_right_on_a_dense_grid_not_just_this_ladder` | `wrong >= 10` on the 60-row dense grid.  `right >= 20`, "no correct row is ever refused" and "every returned wrong row is warned" stay unconditional. |
| `test_fix_pmmstack_sliver_round3.py` (truncation-note truth) | "a truncation-noted row exists in the box subset".  "the round-2 sentence D-5 falsified is gone from every warning" stays unconditional. |
| `test_fix_pmmstack_sliver_round4.py` (move / wall-ratio separation) | `len(wrong) >= 4` and `len(attributed) >= 4`.  `len(right) >= 4` and "no CORRECT row reaches either bar" stay unconditional. |
| `test_fix_pmmstack_sliver_round4.py` (reading-independence ladder) | `small >= 1`, the row that exercises the RETURN side.  Every row's verdict is still asserted reading-independent and on the right side of the arbiter's bars. |

`test_fix_rcwa_even_sector_wsl.py` and `test_verify_rcwa_even_sector.py` were
inspected and carry no assertion of this shape: their claims are on the band's
action and on mode counts of fixed spectra, not on a pathology's magnitude.

---

## 5. The census: from outcome equality to rule conformance

`validation/probe_ci_kernel_sweep/probe_decisions.py` now records, beside each
decision, the **answer class** that decision is about -- measured **with the
guard disarmed**, so it is a property of the arithmetic and not of the verdict:

```
correct   |R+T - 1| < 1e-5      grey   in between      wrong   |R+T - 1| > 1e-2
```

Both bars are placed by the gap, not by a residual: the correct population
reads 1.02e-08 (WIN Haswell) .. 3.66e-07 (CI) and the wrong population 1.17
(Katmai) .. 3.62, leaving five empty decades between them.

`_RULES` states, once, what each family's decision may be for each class:

| row | `correct` | `grey` | `wrong` |
|---|---|---|---|
| `pmm1d_interface/answer@` | `closes` | `open` | `open` |
| `pmm1d_interface/warned@` | `silent` | `silent` or `warn` | `warn` |
| `pmm1d_interface/returns@` | `return` | `return` | `return` |
| `sliver/pmm1d@` | `return` or `warn` | any | `warn` or `refuse` |

`tests/unit/test_ci_kernel_consistency.py` then asserts:

1. **per arm** -- every answer-following decision is one its class permits.
   This is the user-facing claim: a correct answer is never refused, a wrong one
   never returned in silence, on any arm;
2. **across arms** -- rows taken at the **same** class must take the same
   decision.  A row whose decision differs while the class does not, or a
   kernel-independent row (band, branch-cut, mortar, T22 -- no class at all)
   that differs, is still a P1;
3. **rows whose class differs between arms are REPORTED, not failed** -- printed
   with both classes.  That is the guard following the answer;
4. the arm running now is compared **only against census rows of its own
   class**, and its own rule conformance is asserted unconditionally.

A new test, `test_the_census_carries_the_ci_arm_that_answers_these_fixtures_correctly`,
pins the finding itself: the census must carry the CI arm, marked as a
transcription, and the measured arms must still read the 1e-05 sliver row
`wrong` while it reads `correct`.  If the local arms ever start answering that
row correctly, that test fails -- and the failure is the gate working, because
every premise gate in this round cites the divergence.

### The synthetic CI arm

`validation/probe_ci_kernel_sweep/arms/ci_RUNNER_t1.json`, built by
`make_ci_arm.py`, key `CI-unknown-t1`, `"synthetic": true`, with a **per-key
`provenance_detail`**:

* the eight `pmm1d_interface/*` and `sliver/*` rows and their two `rcond` /
  `R+T` readings are **transcribed** from the `test_the_plain_1d_interface_...`
  and `test_this_arm_agrees_...` failure messages (py3.11 shard 4);
* the `band/*` and `branch_cut/*` rows are **inferred**: the consistency test
  re-took the four cheap sections on that arm and named **exactly three**
  disagreeing rows, so every other cheap row equalled the committed consensus.
  They are the kernel-independent sections, so the agreement is expected.
* the mortar and T22 sections are **absent** -- CI never re-takes them.  The
  "every arm answered every row" check is therefore restricted to measured arms.

`numpy` and `scipy` are recorded as `unknown` with the reason, per section 2.2.

---

## 6. The local ladder

Six families, 18 files, 231 tests:

```
tests/unit/test_ci_kernel_consistency.py        tests/unit/test_verify_branch_cut_round2.py
tests/unit/test_fix_branch_cut_round2.py        tests/unit/test_verify_pmm2d_mortar_round3.py
tests/unit/test_fix_pmm2d_mortar_round2.py      tests/unit/test_verify_pmm2d_mortar_round4.py
tests/unit/test_fix_pmm2d_mortar_round3.py      tests/unit/test_verify_pmmstack_sliver_round2.py
tests/unit/test_fix_pmm2d_mortar_round4.py      tests/unit/test_verify_pmmstack_sliver_round3.py
tests/unit/test_fix_pmmstack_sliver_round3.py   tests/unit/test_verify_pmmstack_sliver_walls.py
tests/unit/test_fix_pmmstack_sliver_round4.py   tests/unit/test_verify_rcwa_even_sector.py
tests/unit/test_fix_pmmstack_sliver_walls.py    tests/unit/test_m1_conditioning_guard.py
tests/unit/test_fix_pmmstack_sliver_walls_round2.py
tests/unit/test_fix_rcwa_even_sector_wsl.py
```

Run per arm as
`OMP_NUM_THREADS=n OPENBLAS_NUM_THREADS=n MKL_NUM_THREADS=n OPENBLAS_CORETYPE=k
python -m pytest <the 18 files> -q -p no:randomly -rs`, with the tree pinned by
`PYTHONPATH` and `lumenairy.__file__` verified to be the worktree (an editable
install on this box points at a different clone).

| build | kernel | threads | result | skips |
|---|---|---|---|---|
| WIN py3.14.6 numpy 2.4.4 | Haswell | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WIN | Sandybridge | 1 | **220 passed, 2 skipped** | mortar bit-identity; **M1 X-1 at 7.441x** (M = 22 TE, `sum(R)` = 1.767602e-03, closure 1.558e-03) |
| WIN | Nehalem | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WIN | Katmai | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WIN | Haswell | 4 | **221 passed, 1 skipped** | mortar bit-identity |
| WSL py3.12.3 numpy 2.4.6 | Haswell | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WSL | Sandybridge | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WSL | Nehalem | 1 | **221 passed, 1 skipped** | mortar bit-identity |
| WSL | Katmai | 1 | **220 passed, 2 skipped** | mortar bit-identity; **M1 X-1 at 7.300x** (M = 27 TE, `sum(R)` = 1.738166e-03, closure 1.529e-03) |
| WSL | Haswell | 4 | **221 passed, 1 skipped** | mortar bit-identity |

**Zero failures on ten arms.**  Two arms exercise the gates for real and they
are worth reading twice:

* the mortar bit-identity check **skips on every arm**, naming the two LAPACK
  builds it found and reporting that all three unconditional claims passed
  (worst relative residual 2.566e-15 against a 1.023e-12 bar, worst
  `|guarded - numpy|` exactly 0).  That is the finding of section 2.4 stated as
  a measurement in the log;
* the **M1 X-1 arm reproduces only 7.441x on WIN Sandybridge and 7.300x on WSL
  Katmai** against 152.6x on Haswell -- on this box, two kernels down, the
  engineered pre-fix defect is already an order of magnitude weaker.  The gate
  fires and the test skips with the reading.  The claim that survives on those
  arms is the one the test's name is about: on the row the pre-fix body DID
  break, the withdrawn refusal still moves no bit, and that is asserted before
  the gate.

### The census sweep

Twelve arms: ten measured (`{HASWELL, SANDYBRIDGE, NEHALEM, KATMAI}` x `t1` on
both builds, plus `HASWELL` x `t4` on both), one **unpinned** arm, and the
transcribed CI arm.

```
python validation/probe_ci_kernel_sweep/probe_decisions.py --out arms/<b>_<k>_<t>.json
python validation/probe_ci_kernel_sweep/make_ci_arm.py
python validation/probe_ci_kernel_sweep/merge_arms.py      # 12 arms
```

The unpinned arm matters and is the reason it was taken: **CI's fast lane does
not pin BLAS**, so an argument that the divergence is a thread-count effect has
to be answered by an unpinned measurement, not by a `t4` stand-in.  Measured
`WSL-Haswell-tauto`, 24 BLAS threads, no caps set:

| | `R+T` at 1e-04 | `R+T` at **1e-05** | class at 1e-05 | sliver decision |
|---|---|---|---|---|
| WSL unpinned (24 threads) | 0.9999998749 | **2.1728466467** | **wrong** | **refuse** |
| WSL Haswell t1 | 0.9999998749 | 2.1728583196 | wrong | refuse |
| CI runner (transcribed) | 1.0000003658 | **1.0000010472** | **correct** | **return** |

So the unpinned configuration -- CI's own -- reads it wrong here, to six
significant figures of the pinned arm.  **The thread axis is not the
explanation.**

All twelve arms conform to `_RULES`, and the only rows that differ between arms
are the eight the CI arm computed at a different class.  They are reported by
the gate, not failed.


---

## 7. Skips expected on CI

On an arm that answers these fixtures correctly, the gates fire.  That is the
design, and it is what the matrix should show rather than red:

| test | expected on the CI arm |
|---|---|
| M1 X-1 | SKIP at the 10x gate (CI reproduced 0.267x / 0.677x) |
| verify walls -- remedy slope | SKIP: the guard returns the un-snapped 3e-05 row |
| verify walls -- thin-feature exemption | SKIP: degree spread 1.1315 |
| verify round3 -- R3-A | SKIP: no mount is wrong-as-returned and right-when-snapped |
| verify round3 -- false refusal | SKIP: fewer than two directed steps read CORRECT on pol 1 |
| verify round2 -- within-layer silence | SKIP: no silent-and-broken pair |
| verify branch-cut -- spacer detune | SKIP: the pre-round-1 arm is flat |
| fix walls_round2 -- per-layer window | SKIP: every row correct and silent |
| fix walls_round2 -- LC sliver | SKIP: all three classes right |
| fix mortar_round2 -- plain 1-D | SKIP: the 1e-05 row is correct |
| fix mortar_round2 -- bit-identity | SKIP: numpy and scipy on different LAPACK builds (**also skips locally**) |
| census consistency | PASS -- rule conformance and class-aware comparison |
| fix branch-cut -- spacer ladder pre-arm | SKIP likely (same mechanism as the detune arm) |
| fix walls -- dense grid | SKIP likely: fewer than 10 WRONG rows |
| fix round3 -- truncation note | SKIP possible: no arbitrated row above the super-unity bar |
| fix round4 -- move/wall separation | SKIP likely: fewer than 4 WRONG arbitrated rows |
| fix round4 -- return side | PASS expected (an arm that answers correctly has agreeing rows) |

A skip here is a **measurement in the log**: each carries the reading that made
the premise absent, so the next reader can see how far from the gate the arm sat
without re-running anything.

---

## 8. Open items

1. **WHY the CI arm's arithmetic differs.**  Section 2.  Not solved here.  The
   next step is cheap and is not taken in this round because it needs a CI run:
   add a step that prints `numpy.show_config()`, `scipy.show_config()`,
   `/proc/cpuinfo`'s model name and `OPENBLAS_*` to the matrix log, and install
   `threadpoolctl` on the runner so the census arm can identify its own kernel
   instead of reporting `unknown`.
2. **`threadpoolctl` absent on CI** makes `set_blas_threads` inert there (the
   runner says so itself) and makes the census arm anonymous.  Worth adding to
   the `dev` extra.
3. **The `WSL` label for any linux build** in `probe_decisions._arm_id` is why
   the CI arm reported as `WSL-unknown-t1`.  Cosmetic, but it cost a minute of
   reading; left alone this round because renaming it would invalidate every
   committed arm key.

# CHORE -- the numpy floor, and the two build-dependent tests that had to go first

Branch `chore/numpy-floor` off `origin/main` @ `b7be325`.  Two scoped items, in
order: harden the two per-build test failures
`BUILD_PMM2D_SLANT_METRIC_2026_08_16.md` S15 recorded, then raise the packaged
numpy floor and sweep the consequences.

Binding law throughout: `docs/TESTING_STANDARDS.md`.

## 0. Builds

| tag | what | where |
|---|---|---|
| **W** | Windows 11, py 3.14.6, numpy 2.4.4 / scipy 1.17.1 (scipy-openblas 0.3.31) | native, `C:/tmp/lum_np` |
| L | WSL, py 3.12.3, numpy 1.26.4 / scipy 1.11.4 | `/tmp/venv-floor` |
| CI | WSL, py 3.12.3, numpy 2.5.1 / scipy 1.18.0 | `/tmp/venv-ci` |
| 311 | WSL, py 3.11.15, numpy 2.4.6 / scipy 1.17.1 | `/tmp/venv-py311` |
| 310 | WSL, py 3.10.20, numpy 2.2.6 / scipy 1.15.3 | `/tmp/venv-py310` |

**W is the build both tests failed on**, so green on W is the acceptance.  Every
verification below is `PYTHONPATH`-pinned to the worktree with the import path
asserted, and every run sets the OMP/OPENBLAS/MKL caps before python starts.

---

## 1. [S4] `test_v5_14_0_pmm2d_conical.py::test_conical_jones_matches_rcwa`

### The reading

`assert abs(R1[row].sum() + T1[row].sum() - 1.0) < 3e-2` at `degree=9,
n_orders=4`, 40 deg conical on an anisotropic cell.  On W it read 0.0459.

### What it was actually measuring

That operating point is inside the instability the library warns about at that
exact call ("the truncation is numerically unstable here and the PER-ORDER
efficiencies are suspect").  Code and geometry FIXED, varying only the thread
cap on **one** build (W):

```
threads   |R+T-1| row 0   row 1
1         8.007e-03       2.083e-02
2         1.520e-02       3.038e-03
4         4.589e-02       4.655e-02    <- the recorded failure
8         _EnergyError raised (sum R+T = 2.203)
```

A bar at 3e-2 with a spread running 3.0e-03 -> a hard raise is not measuring the
solver; it is sampling one machine's round-off.

`stabilize=True` does **not** repair it, and this is worth recording because it
is the obvious first idea: the retry ladder treats the 1e-6 closure warning as a
failed attempt, no rung near this geometry clears 1e-6, so it falls back to the
same reading -- measured byte-for-byte identical to the raw one on CI (2.5.1),
L (1.26.4) and 311 (2.4.6).

### The restatement

The hybrid's closure here is TRUNCATION-limited, and *which* rung closes best is
exactly the per-build fact; that a rung closing to ~1e-4 EXISTS is not.  So:

* a stated ladder, `_CONICAL_LADDER = ((11,5,5,3), (13,6,6,5), (15,6,6,5))`
  -- `(pmm degree, pmm n_orders, rcwa n_orders, rcwa upsample)`; the upsample is
  tied to the oracle's `>= 4 n_orders + 1` sampling rule, not chosen;
* the **cross-suite agreement is asserted on every rung** (order-0 `T` against
  the independent RCWA oracle, and the full Jones matrix);
* the **closure bar applies to the rung this build's own measurement selects**.

### The measured envelopes (2026-08-16, 12 build x thread-cap cells)

W at 1/2/4/8 threads; CI, L, 311, 310 at 1 and 4 threads.

| quantity | min | max | bar | gap |
|---|---|---|---|---|
| closure at the SELECTED rung | 1.8e-06 | 9.7e-05 | 1e-3 | 10x below, >1.5 decades to the unconverged reading (4.6e-2 / raise) |
| `|T1[0,0] - T2[0,0]|`, every rung | 2.4e-05 | 3.8e-03 | 3e-2 (unchanged) | ~8x |
| `max|J1 - J2|`, every rung | 2.6e-04 | 3.9e-04 | 5e-2 (unchanged) | ~128x |

Both cross-suite bars keep their shipped values -- they were never the failing
ones and they still clear their envelopes.  Only the closure bar moved, and it
moved because its operating point did.

### Verification

Whole file green on all 12 cells (S4 below).  On W @4 threads the selection
printed `[(11, 5, 0.0001371, 0.0002631), (13, 6, 9.656e-05, 0.0003492),
(15, 6, 6.356e-05, 0.0003518)] selected degree=15 n_orders=6`.

---

## 2. [S2] `test_pmm_m3_efficiency.py::test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build`

### The reading

The test switched OFF the 2026-08-08 passivity widening and asserted that the
scanned family still contains a silent-wrong cell.  On W, with the widening off,
every one of the twelve scanned cells read `rel` = 0.0044-0.0050 with
`n_grow_post` = 0:

```
scanned: [(2,10,0.005), (2,12,0.0047), (2,14,0.0046), (2,16,0.0045),
          (6,10,0.005), (6,12,0.0047), (6,14,0.0045), (6,16,0.0045),
          (2,18,0.0045), (2,20,0.0045), (6,18,0.0044), (6,20,0.0044)]
```

The assertion's own message asked the right question ("If it was FIXED AGAIN,
re-pin this against the fix") and had no build-free answer.

### What it was actually measuring

The narrow 2026-08-06 mask additionally required the mode to sit inside
`_MODE_CUT_MARGIN_WARN` of the cut.  Whether a given build's round-off puts the
mode inside that decade is BLAS reduction order -- so "the narrow mask alone
still leaves a wrong cell" was a per-build fact asserted as a universal one.
Textbook S2.

### The restatement -- a ladder whose last rung is ENGINEERED

`_T34_LADDER`, coarsest (least invasive) first.  The first two rungs are the
library's own shipped switches, so they reproduce the historical defect with no
test-side construction at all:

1. `PMM_FORWARD_GROWTH_PASSIVE = False` -- the 2026-08-06 mask;
2. `PMM_FORWARD_GROWTH_REPAIR = False` as well -- the bare historical selector;
3. `_t34_adverse_flip` -- **the round-off coin forced adverse**.

Rung 3 takes the shipped selector's answer and, for the first `_T34_ADVERSE_N`
(= 1) modes satisfying the defect's OWN scale-free precondition on this build --
`prop AND |Im q| > _MODE_GROWTH_REL |q|`, i.e. classified propagating yet
carrying no z-power, so the flux that classified them is round-off -- takes the
branch that GROWS along +z.  Every injected quantity is read off the solve's own
eigenvalues; there is no constant and no reading from another machine.  Where a
grid has no such mode the array is returned untouched.

`_forward_growth_flip` is a module-global lookup at both call sites and
`_record_mode_cut(..., flip=flip)` records the array the site actually used, so
`n_grow_post` -- the residual channel A the guard fires on -- scores the
INJECTED forward set, not a re-derivation of it.

Structure of the test:

* the two natural rungs are walked first; the first that still reproduces the
  defect **is asserted** (claims b/c/d exactly as before).  A rung that produces
  no wrong cell is **reported, not failed** -- that is this build's shipped mask
  working;
* **rung 3 is unconditional**: it is what makes the test build-free, so it is
  exercised on every build rather than held in reserve;
* claim (e), THE CURE, is made once against the union of scanned cells with the
  shipped selector, unconditional on every arm.

`assert wrong` can now only fire when the *whole* ladder is exhausted, which the
engineered rung makes structurally impossible; its message says what to check if
it ever does (the injector no longer reaching the selector).

Bars are unchanged: same 5 % partition, same 20x separation requirement, same
`1e-5` closure bar.  `_T34_ADVERSE_N = 1` is chosen from measurement, not taste:
over the whole scanned family on W the injected `|R+T-1|` stays at
4.1e-09..5.3e-07 at 1 and reaches 2.2e-06 at 2, so the minimal injection keeps
the SILENT half of "unitary but wrong" farthest from its bar.

### What the rungs produce (4 threads, `_T34_FAMILY` = 8 cells)

Wrong cells / `rel` range against the RCWA anchor:

| rung | W (np 2.4.4) | L (np 1.26.4) | CI (np 2.5.1) |
|---|---|---|---|
| 1 widening off | **0 of 8** | 2 of 8, 0.4399 | **0 of 8** |
| 2 repair off as well | 6 of 8, 0.418-4.958 | (not reached) | 7 of 8, 0.418-4.663 |
| 3 adverse coin | 7 of 8, 0.418-0.440 | 8 of 8, 0.418-4.115 | 8 of 8, 0.418-7.550 |
| cure (shipped selector) | 0 wrong, `post` == 0 everywhere | same | same |

This is the S2 defect, printed: the rung the old test stood on is populated on
exactly one of these three builds.  The engineered rung is the only one
populated everywhere, which is why it is the unconditional one.

### The removed fixture

`passive_widen_off` is gone, replaced by `t34_selector`, which hands the test a
setter for all three knobs and restores all three on the way out.  It had one
user.

---

## 3. The numpy floor

### pyproject

```
numpy>=1.20   ->  numpy>=2.0
scipy>=1.7    ->  scipy>=1.13
```

Mirrored in `requirements.txt` and `requirements-gui.txt` (the latter carried
only the scipy line).  `requires-python` is UNCHANGED at `>=3.10`.

**Adjudication of the scipy floor.**  1.13.0 is the first SciPy line built
against the numpy 2.0 ABI; earlier lines are compiled against the 1.x C-API and
abort at import under numpy 2, so any `scipy < 1.13` is un-installable with this
numpy floor -- the pair has to move together or the metadata lies.  Consistency
with `requires-python`: numpy 2.0 and scipy 1.13 both support Python >= 3.9, and
on 3.10 the resolver lands on the numpy 2.0-2.2 lines, which still ship 3.10
wheels (numpy 2.3 raised its own floor to 3.11, which is why the *floor* and not
a *pin* is the right instrument).

**Why the old floor was a fiction.**  Not a preference -- three independent
reasons, each checkable:

* `np.trapezoid` (added in numpy 2.0; `np.trapz` was REMOVED in 2.0) is called
  directly in `validation/oracles/caustic_fold_truth.py` and
  `validation/oracles/debye_oracle_v3.py` with no fallback anywhere in the
  repo.  Those are the truths the accuracy campaigns are graded against; under
  1.x they raise `AttributeError`.  (They are repo-internal -- packaging is
  `include = ["lumenairy*"]` -- so this is a "the project needs numpy 2"
  argument, not an "import of the wheel fails" one.)
* The library's own dtype reasoning is written AGAINST NEP-50 weak-scalar
  promotion, not around it.  `elements/_lens_real.py` cites the NEP-50 rule by
  name at three sites (`_tf_sl`, the `_obl_p0_src` bookkeeping, the prepared
  real-lens screen) to justify Python-float casts, each with a measured
  ~5e-6 field divergence at `sag_dtype='float32'` when the rule is violated.
  Under 1.x value-based casting that reasoning does not describe the
  arithmetic actually performed, and the halo suites
  (`test_obl_banded_halo.py`, `test_tf_banded_halo.py`) pin it.
* Every CI leg, every local mount and every measured envelope in the suite
  runs 2.x.  Nothing has been run against 1.x in years, so `numpy>=1.20` was
  an untested claim in shipped metadata.

### (a) Workflows -- nothing to change

`grep -rn numpy .github/workflows/` returns six hits, all prose in comments
(BLAS threading notes, a stub-surface list, a "numpy end-to-end" deselect
rationale).  There is **no** job pinning `numpy<2`, no constraints file, no
`--resolution lowest` / oldest-deps leg, and no floor-environment install: every
install line is `pip install -e ".[fft,perf,numba,hdf5,zarr,dev]"` (plus `,jax`
on the parity leg).  `dep-drift.yml` fetches PyPI metadata and asserts each dep
supports 3.10-3.13; it reads the dep LIST from `pyproject.toml` (not the
floors) and needed no edit.  **Nothing was updated or removed here.**

`python scripts/check_dep_metadata.py` was run against the edited
`pyproject.toml` (2026-08-16): `numpy` (latest 2.5.2), `scipy` (1.18.0),
`matplotlib`, `psutil` all report `DRIFT? no`.  The one warning it emits --
`jax` 0.11.0 having moved to `requires-python >= 3.12` -- is PRE-EXISTING,
concerns an optional group, and is untouched by this change.

### (b) Version-conditional numpy code -- the complete list

Swept with `grep -rn --include='*.py'` for `np.__version__`,
`numpy.__version__`, `NumpyVersion`, `importlib.metadata` numpy lookups,
`NEP.?50`, `trapz`/`trapezoid`, the removed-alias family (`np.float_`,
`np.alltrue`, `np.in1d`, `np.row_stack`, `np.product`, `np.NaN`, `np.Inf`,
`np.object_`, `np.bool8`, ...), `np.core`/`numpy.core`, `np.exceptions`,
`hasattr(np, ...)`/`getattr(np, ...)` probes, and free text
(`numpy 1.x`, `older numpy`, `numpy<2`, `legacy numpy`, `pre-2.0`).

**Findings: exactly one artifact, and it is a test, not a library branch.**

| # | what | disposition |
|---|---|---|
| 1 | `tests/unit/test_audit_w5_elements_misc.py::TestP2_07_DammannPhaseProjectorVersionIndependent::test_output_invariant_under_numpy1_sign_semantics` (plus its `_legacy_sign` helper and the `_ShimNp` monkeypatch of `_doe.np`) | **REMOVED.**  It shimmed `np.sign` back to numpy 1.x complex semantics (`sign(z.real)`, falling back to `sign(z.imag)`) and asserted the Dammann design was bit-identical.  No installable numpy has those semantics under the new floor, so the claim is about nothing.  The class docstring records what it was and why it went. |

Deliberately **not** removed, with the reason stated:

* `lumenairy/elements/doe.py` -- the audit-P2-07 explicit unit phasor
  (`z / hypot(re, im)`, 0-safe) is not a version branch; it is what the Octave
  port means, and it is bit-identical to numpy 2.x complex `np.sign` (A/B
  verified when it was written).  Replacing it with `np.sign` would trade a
  correct, self-describing expression for an equivalent one while touching a
  numeric path for no gain.  The comment was rewritten: the numpy-1.x hazard is
  now **excluded by the floor**, not defended against.
* What the removed test protected is still gated one level up and version-free
  by the same class's `test_design_quality_is_sane` (the 1.x degeneration drove
  8x8 uniformity from 0.97 to ~0.02; the gate asserts > 0.5).
* `np.__version__` appears in `validation/` scripts only as printed provenance
  in result headers -- no control flow.  Left alone.

There were **no** `numpy.__version__` conditionals, NEP-50 conditionals, or
`trapz`/`trapezoid` shims in `lumenairy/` at all.

### (c) The WSL floor venv

`/tmp/venv-floor` (py 3.12.3, numpy 1.26.4 / scipy 1.11.4) **no longer
represents a supported floor.**  From this change it is a *historical
divergence mount*, not a floor proxy: it is still the most useful second LAPACK
in reach and it is still the build several campaign envelopes were measured on,
so it stays and it is still worth running -- but a green there proves nothing
about the floor any more, and a failure there is not a supported-configuration
failure.  The build that now represents the floor is a numpy 2.0-2.2 line on
Python 3.10, of which `/tmp/venv-py310` (numpy 2.2.6 / scipy 1.15.3) is the
closest mount available.  **`/tmp/venv-floor` is NOT deleted.**

### `.test_durations`

Three surgical edits, no reformatting (the file is CRLF and only partially
sorted; a `json.dump` round-trip moves ~60 unrelated lines, so it was patched
line-wise and re-validated as JSON -- 12,204 entries):

* the removed numpy-1 test's entry, deleted;
* `test_t34_guard...` 25.45 -> 18.3 s (the ladder replaces a widen-scan with a
  third rung and is *faster* on W);
* `test_conical_jones_matches_rcwa` 0.101 -> 4.0 s (three PMM rungs and two
  RCWA oracles instead of one of each).

The conical entry is the one that mattered: leaving a 0.1 s weight on a 4 s
test is exactly the mis-weighting `AUDIT_CI_TEST_TIME_2026_08_03` S2 records as
the mechanism behind the two publish-verify shard timeouts.

### (d) Verification runs

Full unit suite (`pytest tests/unit -m "not integration"`) on build W --
Windows py 3.14.6, numpy 2.4.4, OMP/OPENBLAS/MKL = 4: see S4.

Targeted pmm / eme / lens suites on `/tmp/venv-ci` (numpy 2.5.1): see S4.

---

## 4. Results

### The two hardened tests, both files, every mount

`tests/unit/test_pmm_m3_efficiency.py tests/unit/test_v5_14_0_pmm2d_conical.py`
= 55 tests.

| build | threads | result |
|---|---|---|
| W (numpy 2.4.4, py3.14) | 1 | 55 passed |
| W | 2 | 55 passed |
| W | 4 | 55 passed |
| W | 8 | 55 passed |
| CI (numpy 2.5.1) | 1 / 4 | 55 passed / 55 passed |
| L (numpy 1.26.4) | 1 / 4 | 55 passed / 55 passed |
| 311 (numpy 2.4.6) | 1 / 4 | 55 passed / 55 passed |
| 310 (numpy 2.2.6) | 1 / 4 | 55 passed / 55 passed |

12 (build x thread-cap) cells, all green -- including the four W cells, which is
the acceptance, and including the 8-thread cell where the OLD conical operating
point raised `_EnergyError` outright.

### Full unit suite and targeted WSL suites

Recorded in S5.

## 5. Run log

### 5.1 Targeted pmm / eme / lens suites on CI (`/tmp/venv-ci`, numpy 2.5.1)

69 files (`ls tests/unit/test_*.py | grep -iE "pmm|eme|lens"`), 4 BLAS threads,
`PYTHONPATH` pinned and asserted:

```
3 failed, 1511 passed, 13 skipped, 2 deselected, 192 warnings in 3300.94s (0:55:00)
```

**All three failures are PRE-EXISTING and are NOT caused by this change.**  Each
was re-run in isolation on BOTH the branch worktree and a worktree standing at
`origin/main` @ `b7be325` (`C:/tmp/lum_sl`, clean, untouched by this branch), on
the SAME interpreter and BLAS cap -- one axis varied, the code:

| test | branch | `origin/main` @ b7be325 |
|---|---|---|
| `test_eme_2d_vector.py::test_vector_structured_completeness` | FAIL | FAIL |
| `test_pmm_m2_window_contract.py::test_min_feature_threshold_rule_predicts_stationarity` | FAIL | FAIL |
| `test_v5_20_13_pmm_jones_2d_fff_nv.py::test_pmm_fff_nv_matches_rcwa_fff_nv` | FAIL | FAIL |

-- identical assertions and identical numbers on both sides (`3 failed in 33.50s`
vs `3 failed in 31.67s`).  None of the three files is touched by this branch, and
`test_pmm_m2_window_contract.py` sorts BEFORE `test_pmm_m3_efficiency.py` under
`-p no:randomly`, so no fixture-state leak from the reworked T3-4 test can reach
it either.

Their signatures, and they are the SAME FAMILY this campaign's two items belong
to -- per-build knife edges, one of which diagnoses itself in so many words:

* `test_vector_structured_completeness`: *"the finder MISSED 1 oracle mode(s)
  that its own condition accepts -- (oracle, converged zero, gaps.min):
  [(77.46594446013133, 77.30034998304453, 1.2974056195835521e-11)]"* -- an S5
  mode census on a 1.3e-11 gap;
* `test_min_feature_threshold_rule_predicts_stationarity`: *"1.5 nm is ABOVE the
  ns=3 threshold"* -- an S4 at-threshold comparison;
* `test_pmm_fff_nv_matches_rcwa_fff_nv`: *"the RCWA fff_nv REFERENCE is unusable
  on this build: no scanned truncation both conserves to better than 1e-03 and is
  reproduced by another one that does, so there is nothing for the PMM to be
  measured against.  This is a statement about the reference, NOT about
  pmm_jones_2d."* -- the scanned closure ladder reads M=7 2.511e-02, M=9
  4.614e-03, M=11 1.001e-02, M=13 4.357e-03, M=15 1.430e-04, i.e. NON-monotone,
  which is the same truncation-instability mechanism S1 above diagnoses on the
  conical Jones cross-check.

**RECORDED, NOT FIXED** -- all three are outside this campaign's scope (it was
scoped to the two tests `BUILD_PMM2D_SLANT_METRIC_2026_08_16` S15 names), and
fixing a fragile test without measuring its cross-build envelope first is the
mistake `TESTING_STANDARDS` exists to prevent.  Flagged here because they are the
same shape, they are on `origin/main` today, and the third one is very close in
kind to S1 -- a follow-on wave should take all three together with the same
method.

### 5.2 Full unit suite on build W

`pytest tests/unit -m "not integration"`, Windows py 3.14.6 / numpy 2.4.4 /
scipy 1.17.1, OMP/OPENBLAS/MKL = 4, `PYTHONPATH` pinned and asserted (`PIN OK
C:\tmp\lum_np\lumenairy\__init__.py`).

(pending -- still running at the time this section was written; result and, for
any failure, the same one-axis `origin/main` comparison as 5.1, to be appended)

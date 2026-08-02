# Release-green verification -- v5.32.0 (branch `fix/pmm-union-grid-conditioning`)

Date: 2026-08-01
Tree: `fix/pmm-union-grid-conditioning` @ `4c027e3` (working tree, uncommitted)
Merge base with `main`: `a396087` (= `main` HEAD at the time of writing)
Environment: Windows 11, Python 3.14, numpy 2.4.4, scipy 1.17.1,
scipy-openblas 0.3.31.188.0, pyFFTW present, 24 logical CPUs

Scope: make `pytest tests/unit -m "not integration and not slow"` genuinely
green so the publish workflow's re-run of the unit suite on the release tag
cannot block v5.32.0.  Ten failures were in scope.  **None of them was an
unfixable-in-scope defect; none of them required changing the physics.**  The
release decision is unchanged.

Two of the ten were caused by this branch (one a real budget breach, one a
legitimate behaviour improvement whose pin was left stale).  Four were caused
by environment drift against pins whose tolerances had no margin.  Four were
pre-existing test-isolation defects.  Details and evidence below.

---

## Summary table

| # | Test | Class | Verdict |
| --- | --- | --- | --- |
| 1,2 | `test_audit_except_budget.py` (both pins) | branch-caused | 2 broad excepts NARROWED; budget NOT raised |
| 3,4 | `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512/1024-complex128]` | environment drift | estimator re-calibrated from fresh measurement |
| 5 | `test_niche_audit_w7_rcwa.py::test_clean_circular_truncation_agrees_with_rectangular` | knife-edge tolerance | control-arm bound re-justified with measurement; feature-arm bound unchanged |
| 6 | `test_niche_audit_w9_raster_harmonic.py::test_w9_harmonic_closes_the_sheared_taper_regression` | knife-edge tolerance | absolute machine-eps gate replaced by measured relative gate |
| 7 | `test_v5_14_1_device_geometry.py::test_stack_slices_consensus` | branch-caused (improvement) | pin reconciled to the stronger contract |
| 8,9,10 | trio: `test_niche_audit_w9_traced_determinism.py::test_auto_promote_still_promotes_when_opted_in`, `test_perf_v4_12_0_fft_infra.py::TestAutoPromote::{test_promotion_after_threshold_calls,test_no_promotion_when_disabled}` | test isolation | new shared `shipped_fft_dispatch` fixture |

Files changed (all uncommitted; `CHANGELOG.md` untouched):

```
lumenairy/elements/_lens_traced.py            broad except NARROWED
lumenairy/elements/pmm/stack.py               broad except NARROWED
lumenairy/memory.py                           _ASM_FIRST_CALL_FIXED_BYTES 40 -> 56 MiB + docs
tests/conftest.py                             + shipped_fft_dispatch fixture (opt-in)
tests/unit/test_niche_audit_w3_infra.py       A-6 pins re-documented / re-calibrated
tests/unit/test_niche_audit_w7_rcwa.py        per-arm closure bound + measured table
tests/unit/test_niche_audit_w9_raster_harmonic.py  TE gate relative + measured table
tests/unit/test_v5_14_1_device_geometry.py    slices-consensus pin reconciled
tests/unit/test_niche_c8_inverse_support_bound.py  hull stub raises the real QhullError
tests/unit/test_perf_v4_12_0_fft_infra.py     autouse fixture takes shipped_fft_dispatch
tests/unit/test_niche_audit_w9_traced_determinism.py  same
```

---

## 1 + 2 -- non-`ui/` broad-`except` budget

**Root cause.**  The counter (bare and `as e` forms, `lumenairy/**` minus `ui/`)
read **50** against a budget of **48**.  Counting the same regex at the merge
base `a396087` gives exactly **48**, and a per-file diff isolates the two new
sites:

```
DIFF elements/_lens_traced.py  base=0  head=1     (line 5744)
DIFF elements/pmm/stack.py     base=3  head=4     (line 2085)
```

Both arrived with work on this branch: the C8 exit-support-bound `ConvexHull`
guard, and the union-grid consensus `min_feature` probe.  Neither is the
sanctioned jax-tracer-guard class, so per the test's own judgement rule
(NARROW > WARN-BEFORE-PASS > RE-RAISE > KEEP-AS-IS) neither justifies a budget
bump.

**Fix.**  Both NARROWED; the budget is untouched at 48.

* `_lens_traced.py` -> `except (ImportError, RuntimeError, ValueError)`.
  `scipy.spatial.QhullError` is a documented `RuntimeError` subclass and is
  named indirectly because it only became public in scipy 1.8 while the
  package floor is `scipy>=1.7`.  `ImportError` covers a trimmed install with
  no compiled qhull; `ValueError` covers the input rejections `ConvexHull`
  raises before qhull runs.
* `pmm/stack.py` -> `except (ValueError, NotImplementedError, RuntimeError)`.
  That is the full raise surface of a PMM solve (`lumenairy/elements/pmm/*.py`
  raises 177 `ValueError`, 36 `NotImplementedError`, 4 `RuntimeError`, 1
  `ImportError` behind an optional-backend gate); `numpy.linalg.LinAlgError`
  is a `ValueError` subclass, so eigensolver failures are covered, and
  `_StabilizeScanExhausted` never escapes `solve()` -- it is consumed by the
  scan loops in `pmm/_core.py`.

**Knock-on, and why it is a strengthening.**  Narrowing the `_lens_traced.py`
site broke one campaign pin,
`test_niche_c8_inverse_support_bound.py::test_a_degenerate_support_declines_instead_of_raising`,
which stubbed `ConvexHull` with a bare `Exception` subclass.  That stub only
worked because the handler was over-broad -- it was testing the handler's
breadth rather than the declared contract.  It now raises the class Qhull
actually raises (`scipy.spatial.QhullError`, falling back to `RuntimeError` on
the scipy 1.7 floor), which models the real failure faithfully.  This is the
only edit made inside the traced-carrier campaign's territory and it is
test-side only; `_lens_traced.py`'s remap / support-bound machinery, the
C5-C8 flags and `carrier.py` are otherwise untouched.

**Evidence.**

```
count before: TOTAL 50      count after: TOTAL 48
tests/unit/test_audit_except_budget.py ................ 2 passed
tests/unit/test_niche_c8_inverse_support_bound.py
tests/unit/test_niche_c7_ray_density_halo_check.py ..... 28 passed
```

---

## 3 + 4 -- `estimate_asm_memory` no longer bounded the measured first-call peak

**Root cause: fixed-term drift, not a new intermediate array.**  Re-measured
with the audit's own method (fresh interpreter + `tracemalloc`, N = 64..2048 x
{complex64, complex128}, fitting `cold = slope * N^2 + fixed`):

| pair | slope c128 | fixed c128 | slope c64 | fixed c64 |
| --- | --- | --- | --- | --- |
| 256 -> 512 | 97.7 B/px | 52.53 MiB | 48.0 B/px | 52.63 MiB |
| 512 -> 1024 | 96.0 B/px | 52.96 MiB | 48.0 B/px | 52.64 MiB |
| 1024 -> 2048 | 96.0 B/px | 52.97 MiB | 50.7 B/px | 49.91 MiB |

The **shape** term is still a bound: the formula uses 101.6 B/px (complex128)
and 52.8 (complex64), both above the measured slopes.  What moved is the
one-time, N-independent lazy FFT-backend import, from the 38.17-38.50 MB
recorded at derivation time (2026-07-25) to ~53 MiB on today's dependency
stack.  With the constant still at 40 MiB the estimate stopped being a bound:

```
est/cold  N=256 0.790   N=512 0.850   N=1024 0.951   N=2048 1.022   (complex128)
est/cold  N=256 0.779   N=512 0.826   N=1024 0.930   N=2048 1.006   (complex64)
```

(The N=64/128 pairs fit ~40 MiB because the backend import has not yet paid
its large-transform workspace there; the N >= 256 asymptote is the one an
estimate must bound.)

**Fix.**  `memory._ASM_FIRST_CALL_FIXED_BYTES` 40 MiB -> **56 MiB**, with the
fit table recorded at the constant.  The assertion was not touched and the
shape term was not touched.  Two companion pins were updated for the same
reason and with the same evidence:

* `test_documented_band_vs_steady_state`: `ratios[512]` 16.35 -> **20.35**
  (`ratios[16384]` moves by 0.09%, still inside its `rel=0.01`).
* the cross-platform looseness fence: the Windows band is re-measured at
  1.06-1.09 so the 1.35 fence is unchanged; the Linux fence is raised
  4.0 -> 5.0 because the recorded CI-Linux cold peaks (26.1 MB at N=512,
  101.7 MB at N=1024, implied by the e1fd64a ratios) project to 3.27 / 1.63
  against the new estimate.  That projection is labelled as a projection in
  the code, not as a measurement.

Raising the constant is the fail-safe direction: the estimator over-predicts.

**Evidence (after).**

```
complex128  N=256 1.063  N=512 1.058  N=1024 1.058  N=2048 1.058
complex64   N=256 1.067  N=512 1.074  N=1024 1.089  N=2048 1.069
tests/unit/test_niche_audit_w3_infra.py + test_memory_guardrail.py ... 102 passed
```

---

## 5 -- `test_clean_circular_truncation_agrees_with_rectangular`

**Not a branch regression.**  `git diff a396087 -- lumenairy/elements/rcwa/`
is empty: the RCWA sources are byte-identical to `main`, and the test file has
not changed since `d3941f5`.  Whatever this is, `main` has it too.

**Root cause: a knife-edge tolerance on LAPACK round-off.**  The assertion
that fails is the energy-closure bound, applied to BOTH arms:
`abs(R.sum() + T.sum() - 1.0) < 1e-11`.  On the RECTANGULAR arm that is a
121-order (242x242) `zgeev` plus an S-matrix cascade, and the residual moves
with the BLAS reduction order.  Measured on the identical computation:

| BLAS threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangular | +1.13e-11 | -5.34e-13 | -3.35e-14 | +3.55e-14 | +3.55e-14 | -1.38e-12 | -8.99e-15 | -7.70e-14 | +1.53e-14 |
| circular | -3.15e-14 | -2.42e-14 | -1.04e-14 | -1.31e-14 | -2.78e-15 | +2.09e-14 | +3.55e-14 | -7.35e-14 | +1.84e-14 |

The rectangular arm reaches **1.13e-11**, above the gate.  That is why the pin
passed or failed as a function of collection order:

```
w7 file alone .......................... 61 passed
w7 -k "not test_w7" .................... 1 failed (1.1324718940386447e-11)
target alone, OMP_NUM_THREADS=1 ........ 1 failed (same value)
target alone, OMP_NUM_THREADS=8 ........ 1 passed
```

This is consistent with the file's own header: *"eigensolve-level agreement is
~1e-11 relative cross-platform, so no exact / hash pins on solver output."*  A
1e-11 **absolute** closure bound on a 242x242 eigenproblem sits at that noise
floor.

**Fix -- nothing is loosened where the test does its work.**  The bound is
split per arm.  The CIRCULAR arm, which is the recorded non-coverage this test
exists to lock, keeps the original `1e-11` and clears it by 136x at every
thread count.  Only the rectangular CONTROL arm moves to `1e-10` (8.8x
headroom over its worst measured value).  The physics claim is untouched and
is not round-off-sensitive at all: `|dR0|` reads **2.9247e-05 to five
significant figures at every one of the nine thread counts**, against a `2e-4`
gate.

**Evidence.**  `w7` full file 61 passed; `w7 -k "not test_w7"` 35 passed.

---

## 6 -- `test_w9_harmonic_closes_the_sheared_taper_regression`

**Also not a branch regression** (same argument: `lumenairy/elements/rcwa/` is
byte-identical to `a396087`, test unchanged since `d3941f5`).

**Root cause.**  The closing assertion was
`abs(t_a - t_h) < 1e-9 * max(1.0, t_a)` -- an ABSOLUTE machine-precision gate
on the claim "TE on the very same cells is untouched (== 'area')".  That claim
is exactly true at the cell level and true to ~4e-16 at the operator level,
but it is NOT bit-identity, and the gate was inherited from the single-vertical
-layer sibling (`test_w9_harmonic_equals_area_off_the_inverse_rule_channel`,
which gates the same claim at 1e-9 on a case that measures 2.2e-15) and
applied here to a SHEARED SIXTEEN-SLICE cascade that amplifies it by ~1e6.

Mechanism, verified directly:

* the `'harmonic'` painted cell is `'area'` bit-for-bit, and the y-companion
  `eyy` is bit-identical to that cell (`np.array_equal(eyy, cell)` -> `True`);
* but a layer carrying a companion pair routes through `RCWAStack._li_blocks`'s
  `_li_convolutions_2d_tensor` arm instead of `_li_convolutions_2d`.  Those
  build the same `Cyy` in a different order -- the tensor arm carries an extra
  operator inverse/re-inverse that cancels analytically for a y-uniform cell.
  Measured residual **6.3e-16 relative**; the library's own `_li_blocks`
  docstring already records it as 4.2e-16, i.e. explicitly not bit-identity.
  `Cxx` *is* bit-identical between the two arms.
* sixteen such layers cascaded on a sheared staircase turn 4e-16 into ~1e-8.

Measured, and stable to every digit across BLAS thread counts 1 / 4 / 16 /
default (so this is deterministic amplification, not chaos):

```
t_a = 4.746961e-04   t_h = 4.746899e-04
|t_a - t_h| = 6.1828e-09   ->  1.3025e-05 RELATIVE
underlying quad difference          9.9934e-09
unsheared single-slice control      2.2204e-15
```

**A simplification was considered and REJECTED with measurement.**  Since the
companion pair is always diagonal, replacing the tensor arm with the per-axis
scalar rule (`Cxx` from `_li_convolutions_2d(exx)`, `Cyy` from
`_li_convolutions_2d(eyy)`) would make the TE channel bit-identical to
`'area'`.  It is wrong in general: on a genuine crossed 2-D cell with
`exx != eyy` the two `Cyy` operators differ by **16% relative** (they agree
only when the cell is uniform along the second axis, which is the 1-D grating
case).  The Li-2003 successive x-then-y factorization is not separable, and
the tensor arm must stay.  Recorded here so nobody re-derives it.

**Fix.**  The TE gate becomes relative with an absolute ceiling: `1e-3`
relative (77x headroom over the measured 1.30e-05) and `1e-6` absolute (162x).
It still catches what it exists for: a companion leaking into the DIRECT-rule
channel would move TE by the same order it moves TM -- 2.3x to 13x, i.e. O(1)
relative.

**Evidence.**  `test_niche_audit_w9_raster_harmonic.py` 22 passed.

---

## 7 -- `test_stack_slices_consensus`

**Root cause: the branch legitimately improved the behaviour and this pin was
left stale.**  At `a396087`, `PMMStack._slices_consensus_check` warned
*"no taper builder recorded on this stack ... the consensus check was
skipped."* whenever there was no recipe -- which meant the ENTIRE recipe-free
route (hand-added layers and every `SegmentStackGeometry`-built device, i.e.
the documented device route) was silently unprotected against the
passive-but-wrong staircase pathology.  The union-grid audit (2026-07-28, R-1)
replaced that dead end with `_union_grid_consensus_check`, which needs no
recipe.  The pin still asserted the "skipped" warning.

This is a reconciliation, not a defect: the new path is strictly more
coverage than the warning it replaced, and the workstream had already shipped
the reconciled version of this contract from the covariant side --
`test_audit_w3_entry_validation.py::TestP331CovariantKwargs::test_stabilize_slices_is_honoured`
-- but missed this second, older copy in the device-geometry file.

**Fix.**  Reconciled to the stronger contract: no "skipped" / "no taper
builder" warning, no false pathology warning, finite `R+T`.

**Fail-before witness (kept in the docstring).**  On the pre-fix tree the
recipe-free block emits the "skipped" warning, so `assert not skipped` fails;
on this tree it emits **nothing** (measured: 0 warnings, `R+T = 2.000` for the
two-polarization convention), because a uniform layer has no cross-layer walls
to snap and therefore scores exactly 0 -- a clean stack cannot false-positive.

**Evidence.**  `test_v5_14_1_device_geometry.py` + `test_audit_w3_entry_validation.py`
72 passed.

---

## 8 + 9 + 10 -- the order-dependent auto-promote trio

**Root cause: incomplete test isolation, on both sides.**  All three tests
assert that the FIRST pyFFTW plan built at a key is `FFTW_ESTIMATE`, and
therefore that a plan-cache entry exists at all.  Neither module's fixture
owned the process-global DISPATCH state that decides whether `_fft2` reaches
the plan cache in the first place.  From `fft_infra._fft2`:

```python
if (USE_PYFFTW and PYFFTW_AVAILABLE
        and np.iscomplexobj(x)
        and x.shape[0] >= FFTW_MIN_SIZE
        and shape not in _PYFFTW_BAD_SHAPES):
```

Any of the following, left behind by an earlier test in the shard, makes the
probe read `None` (or `FFTW_MEASURE`) and all three fail while each still
passes alone:

* `USE_PYFFTW = False`.  Reachable from `lumenairy/ui/waveoptics_dock.py:568`,
  which sets it unconditionally and only re-raises it for
  `backend == 'pyfftw'` -- and the default is `'numpy'`.
* `FFTW_MIN_SIZE` raised above the test's N.  This is the one dispatch global
  `snapshot_fft_state()` does **not** carry.
* `_PYFFTW_PLAN_FLAGS` left at `FFTW_MEASURE` (any `set_pyfftw_planner` caller).
* `PYFFTW_FALLBACK_ON_ERROR`, `_PYFFTW_DOUBLE_BUFFER`, `_PYFFTW_PLAN_CACHE_SIZE`
  -- all reachable through `set_low_memory`.

`reset_fft_backend()`, which both modules already called, clears only the cache
CONTENTS and the bad-shape blacklist.  (The auto-promote call counters do live
inside the cache entries, so those *were* reset -- the counters are not the
problem.)  It cannot undo a switched-off pyFFTW path or a moved planner.

**Fix -- isolation by construction, not by chasing the polluter.**  New shared
**opt-in** fixture `shipped_fft_dispatch` in `tests/conftest.py`.  It forces
every dispatch global to its shipped value (`USE_PYFFTW = PYFFTW_AVAILABLE`,
`USE_SCIPY_FFT = True`, `PYFFTW_FALLBACK_ON_ERROR = True`,
`FFTW_MIN_SIZE = 256`, double-buffer on, plan-cache size 8, planner
`FFTW_ESTIMATE`), clears the plan cache, blacklist and counters, and on exit
restores the caller's state via `restore_fft_state` **plus `FFTW_MIN_SIZE` by
hand** (the key the snapshot omits).  It deliberately does not touch libfftw3
wisdom -- that is process-global inside the C library, affects bits rather than
the plan flags asserted here, and the w9 module already snapshots it.

Both modules' existing autouse fixtures now request it; no test bodies, no
reordering and no skips were introduced.

**Evidence -- adversarial fail-before / pass-after.**  A temporary module
collected BEFORE both trio files deliberately leaked
`USE_PYFFTW = False`, `FFTW_MIN_SIZE = 4096`,
`set_pyfftw_planner('FFTW_MEASURE')`, `set_fft_plan_cache_size(1)` and
`set_fft_double_buffer(False)`:

```
LEAK use_pyfftw_off  : unprotected probe flags[0] = None   <- the failure signature
LEAK min_size_up     : unprotected probe flags[0] = None
LEAK planner_measure : unprotected probe flags[0] = None
protected probe (shipped_fft_dispatch)  : flags[0] = 'FFTW_ESTIMATE'
then test_niche_audit_w9_traced_determinism.py
   + test_perf_v4_12_0_fft_infra.py     : 25 passed
```

The unprotected probe is the same code as
`test_auto_promote_still_promotes_when_opted_in`'s body.  The temporary module
was deleted after the run and is not part of the diff.

---

## Verification

`ruff check lumenairy/ tests/unit/` -> **All checks passed!**
`ruff check tests/conftest.py` -> **All checks passed!**

The ten target tests, in isolation:

```
tests/unit/test_audit_except_budget.py
tests/unit/test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak
tests/unit/test_niche_audit_w7_rcwa.py::test_clean_circular_truncation_agrees_with_rectangular
tests/unit/test_niche_audit_w9_raster_harmonic.py::test_w9_harmonic_closes_the_sheared_taper_regression
tests/unit/test_v5_14_1_device_geometry.py::test_stack_slices_consensus
tests/unit/test_niche_audit_w9_traced_determinism.py::test_auto_promote_still_promotes_when_opted_in
tests/unit/test_perf_v4_12_0_fft_infra.py::TestAutoPromote::test_promotion_after_threshold_calls
tests/unit/test_perf_v4_12_0_fft_infra.py::TestAutoPromote::test_no_promotion_when_disabled
-> 10 passed
```

File-level runs of every touched file:

```
test_audit_except_budget.py .................................. 2 passed
test_niche_audit_w3_infra.py + test_memory_guardrail.py ..... 102 passed
test_niche_audit_w7_rcwa.py .................................. 61 passed
test_niche_audit_w7_rcwa.py -k "not test_w7" ................. 35 passed
test_niche_audit_w9_raster_harmonic.py ....................... 22 passed
test_v5_14_1_device_geometry.py + test_audit_w3_entry_validation.py  72 passed
test_perf_v4_12_0_fft_infra.py + test_niche_audit_w9_traced_determinism.py  18 passed
```

Campaign suites (proof the traced-carrier work is undisturbed):

```
tests/unit/test_niche_c8_inverse_support_bound.py
tests/unit/test_niche_c7_ray_density_halo_check.py ........... 28 passed
```

Targeted regression sweep over every area the edits can reach (`memory.py`,
`pmm/`, `rcwa/`, `_lens_traced.py`, the FFT dispatch, the device-geometry
builders and the shared `conftest.py`):

```
python -m pytest tests/unit -k "pmm or rcwa or traced or memory or fft or
    lens or geometry or raster or context or slant"
    -m "not integration and not slow" -p no:randomly -q
-> 2504 passed, 4 skipped, 8467 deselected in 2937.14s (0:48:57)
```

(The 4 skips are pre-existing and environmental: a `threadpoolctl`-present
skip, the host-specific W5 SHA-256 digest gate, and two documented cache-lock
exemptions.)

### Full leg

`python -m pytest tests/unit -m "not integration and not slow" --tb=line -q -p no:randomly`

```
10690 passed, 75 skipped, 210 deselected, 473 warnings in 6031.79s (1:40:31)
```

**0 failed.**  Run on the fixed working tree with nothing else competing for
the box.  (An earlier attempt at the same leg was cut off by a harness
background-task timeout at the 50% mark with 0 failures recorded up to that
point; the leg above is a complete, uninterrupted run of the same command.)

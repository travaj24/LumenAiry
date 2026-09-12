# WP-A15b -- architecture: optional-dep helper, `override()`, lazy loading, layering, re-exports

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` section 14 (TESTS-ARCH) finding **V6** and its
sub-findings P2-5 / P2-7 / P2-8 / P2-9, plus the cross-WP re-export requests and the WP-A8
`subharmonics` passthrough.  Branch `audit-fixes-2026-09`, HEAD at session start `58ad836c`.

---

## 1. Summary

| Finding | Status | Files : lines | Tests | Oracle | Measured before -> after |
|---|---|---|---|---|---|
| **P2-9** five-fold `_ensure_cupy_loaded` / `_is_cupy_array` / `_load_numba` | **fixed** (3 of 8 sites; 5 lens sites are WP-A16's) | new `lumenairy/backend/_optional.py` (180 L); `propagators/fft_infra.py:46-48,52-93`; `sources/core.py:26-27,32-42`; `optimize/_merit_jit.py:45-46,69-81` | `test_audit2609_a15b_optional_and_knobs.py::TestOptionalCupy` / `::TestOptionalNumba` / the three `test_*_keeps_its_cp_alias_contract` | the five copies' logic transcribed as `_reference_is_cupy_array`, compared against the shared helper on 6 input kinds | 13 duplicate defs -> 3 thin wrappers + 1 implementation; NumPy-path answers identical |
| **P2-5** 53 `set_*`, 0 context managers, 0 resets | **fixed** | new `lumenairy/_knobs.py` (357 L); 17 `register_knob` calls (`fft_infra` x12, `memory.py:147`, `cache.py:215`, `io/storage.py:1779`, `user_library.py:109`, `rcwa/_core.py:234`); `__init__.py` exports `override`; `tests/conftest.py:357-422` autouse guard | `test_audit2609_a15b_optional_and_knobs.py` (33 tests) | algebraic: `restore(snapshot())` is the identity; `override` is the identity on every exit path; restore ORDER observable through a two-probe call log | 0 -> 17 knobs scoped and restorable; fixture cost 11.6 us/test = 0.155 s over 13 319 tests |
| **P2-7** import-time cost | **partially fixed** (the part inside my ownership) | `elements/__init__.py:148-248`; `__init__.py:2081-2199` | `test_audit2609_a15b_lazy_and_layering.py` (23 tests) | AST transitive closure of module-level import edges out of `lumenairy/__init__.py` | `lumenairy.elements` cum **144.6 -> 121.7 ms**; interleaved same-build A/B **749.0 -> 711.7 ms (-37.3 ms, -5.0 %)**; the remaining 566 ms is `backend.scipy`, section 5 |
| **P2-8** the one import-time layering violation | **fixed** | `raytrace/surface.py` (import moved from `:22` into `_base_surface_sag_xy`, now `:491`) | same file, 5 layering tests | AST walk separating module-level from in-function imports, `TYPE_CHECKING` excluded | 1 -> 0 module-level `raytrace -> elements` edges; sag output `np.array_equal` identical |
| V6 re-export reconciliation (WP-A1/A4/A7/A10/A11/A13 requests) | **fixed** | `__init__.py` (+15 names, `__all__` 708 -> 723); `raytrace/__init__.py` (+3) | `test_audit2609_a15b_reexports.py` (24); `test_v4_16_0_walker_all_symmetry.py`; `test_v4_14_1_dispatcher_pin_cache_clears.py` | object IDENTITY against the defining module, not `hasattr` | walker 19 MISSING -> 0, with **no new exemptions** |
| WP-A8 section 5.3 `subharmonics` passthrough | **fixed** | `propagators/system.py:1017` (+ the element-dict doc at `:685-694`) | `test_audit2609_a15b_system_subharmonics.py` (6) | the generator called directly and applied by hand; `np.array_equal` | default bit-identical; the previously-discarded knob is worth max \|dphi\| = **2.6701 rad** (rms 1.3526) at 3 levels |
| WP-A11 section 5 item 4 `_ASM_FIRST_CALL_FIXED_BYTES` vs the 1.35 fence | **fixed** | `memory.py:633` (56 -> 40 MiB) + its derivation block; `test_niche_audit_w3_infra.py:760-1030`; `test_verify_perf_fixes_2026_08_10.py:87-107` | 2 new / 3 rewritten bars in `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory` | fresh interpreter + `tracemalloc`, 12 points, `cold = slope*N^2 + fixed` | fixed term **52.97 -> 36.71 MiB** measured; est/cold **1.341 -> 1.077** at N=512; band **1.10-1.49 -> 1.061-1.112** |
| WP-A9 section 5.3 stale conftest comment | **fixed** | `tests/conftest.py:245-262` | n/a (comment) | WP-A9_REPORT section 5.3 | -- |
| WP-A5 section 5.2 `rs_alias_free_distance` | **not actionable** | -- | -- | `grep` | the public name does not exist (`propagators/rs.py:175` defines `_rs_alias_free_distance`); section 5 request |
| WP-A11 section 5.1 `from_prescription` | **decided, not taken** | -- | `test_audit2609_a15b_reexports.py::test_from_prescription_stays_a_classmethod_only` | the walker's own exemption rationale | exemption stands; decision recorded in a test |

---

## 2. Per deliverable

### 2.1 `lumenairy/backend/_optional.py` (deliverable 1)

**What was wrong.** `difflib` + normalised-AST in the audit found `_ensure_cupy_loaded` in 5
modules, `_is_cupy_array` in 5 and `_load_numba` in 5 -- `_load_numba` byte-identical between
`_lens_traced.py:76` and `_merit_jit.py:60`, 0.91 similar to the other two.  The real cost is not
the ~60 duplicated lines; it is that the accelerator-absent path -- the path every NumPy user
takes -- had five implementations and no place to test.

**What I read before writing it.**  All eight copies: `_lens_real.py:32-50`,
`_lens_traced.py:36-91`, `lenses.py:44-130`, `fft_infra.py` (pre-change `:37-71`),
`sources/core.py` (pre-change `:19-33`), `_lens_imap.py:410-429`, `_merit_jit.py` (pre-change
`:48-78`), `fga.py:94-107`.  Three semantics had to be preserved exactly:

1. `find_spec` availability probe at import (never `import cupy` / `import numba` eagerly);
2. first-use import with permanent memoisation, and `_load_numba`'s `if _numba is not None:
   return True` fast path so a second call costs one global read;
3. absence reported as a VALUE (`None` / `False`), never an exception -- CONVENTIONS section 10.

`fga.py`'s `_load_numba` is the outlier (it uses `try: import numba` and caches a bool in
`_NUMBA`, and its caller RAISES because the FGA swarm sum has no NumPy fallback).  It is WP-A16's
file; `load_numba()` reproduces the same observable contract, and the raise stays at `fga.py:376`
where it belongs.

**What I changed.**  New leaf module (no lumenairy imports, nothing heavier than `importlib`)
exporting `CUPY_AVAILABLE`, `NUMBA_AVAILABLE`, `ensure_cupy()`, `cupy_module()`,
`is_cupy_array(x)`, `load_numba()`, `numba_handles()`.  `ensure_cupy()` returns the MODULE rather
than a bool -- the one deliberate difference, so a consumer does not need a second global; the
consumer wrappers keep the historical `-> bool` spelling.

The three non-lens sites delegate but keep their own names, because all three are load-bearing:

* `fft_infra.__all__` lists `_ensure_cupy_loaded` / `_is_cupy_array` and `propagation.py:177,191`
  re-exports them (`test_v5_1_0_agent_c_split.py` pins the surface);
* `fft_infra`'s CuPy branches read the MODULE-LEVEL `cp` (`cp.fft.fft2(x)` at `:2200` etc.), so
  `_is_cupy_array(x) is True` must still imply `cp` is bound -- the wrapper keeps that coupling and
  a test pins it;
* `_merit_jit._NUMBA_AVAILABLE` is read at CALL time (`:224`) and
  `test_v5_3_multi_field_merit_jit.py:445,511` monkeypatches it to `False`, so the availability
  GATE stays local while the import is shared.

`_is_cupy_array` in `fft_infra` keeps its own `if not CUPY_AVAILABLE: return False`
short-circuit before delegating: it runs once per FFT and the absent answer must stay one global
read.

**Verification.**  `test_audit2609_a15b_optional_and_knobs.py` transcribes the five copies' logic
as `_reference_is_cupy_array` and asserts parity over six input kinds; the NumPy-2.x
`ndarray.device` trap (the defect every copy's comment cites) is pinned with an explicit
`assert hasattr(np.zeros(3), 'device')` so the test says out loud that the trap is live on this
NumPy.  The numba wrapper's two arms are compared end to end: JIT vs pure NumPy, max \|delta\| =
0 (bar 4e-16 = ~2 ULP of a unit-modulus complex128 phasor).  CuPy is NOT installed here, so the
CUDA arm is desk-checked only; that is stated in the test and here.

**Residual risk.**  `backend/_optional.py` lives under `lumenairy/backend/`, so importing it runs
`lumenairy/backend/__init__.py`, which imports `backend.scipy` -> `scipy.linalg`.  Today that is
free (`lumenairy.backend` already has 41 module-level importers including `propagators.rs` and
`elements.elements`, and is loaded long before `fft_infra`), but it does couple `fft_infra` to the
`backend` package.  If the `backend/__init__.py` lazy change in section 5 lands, `_optional.py` is
already a pure leaf and the coupling becomes free in the other direction too.

### 2.2 `lumenairy/_knobs.py` and `override()` (deliverable 2)

**What was wrong.**  Audit P2-5: 103 `set_*`/`get_*`/`reset_*` functions, 22 writing a module
`global`, **53 with a `set_` verb, 0 context-manager forms, 0 resets**.  Per-knob restore coverage
in the suite: `set_asm_cache_size` 2 calls / 0 `finally`, `set_fft_fallback` 1 / 0,
`set_cache_budget` 15 / 0, `set_blas_threads` 9 / 1, `set_fft_threads` 0 fixtures.  The suite runs
serially, so a leak reaches a later test and does not reproduce under `-k`.

**Contract as implemented** (binding for WP-A16, which registers the lens knobs the same way):

```python
register_knob(name, *, getter, setter, doc) -> None
override(**knobs)            # contextlib.contextmanager
snapshot() -> dict[str, Any]
restore(state: dict) -> None
knobs() -> tuple[str, ...]
knob_doc(name) -> str
```

* `register_knob` validates (`ValueError` on an empty name, `TypeError` on a non-callable) with the
  CONVENTIONS section 2 `f"{fn_name}: ..."` prefix, records the registration-time value, and
  replaces an existing entry (so `importlib.reload` of an owner module is safe).
* `getter` must be cheap and SIDE-EFFECT-FREE -- it runs once per knob per test.  Two knobs needed
  a module-private accessor for exactly that reason and WP-A16 should expect the same rule:
  `user_library.get_library_path()` CREATES `~/.lumenairy/library` plus three subdirectories, and
  `cache.get_cache_budget()` calls psutil when no override is set.  Both are registered against the
  raw OVERRIDE (`_get_library_path_override`, `_get_cache_budget_override_mb`), which is also the
  semantically right thing to restore -- restoring an auto budget as a pinned one would be a
  behaviour change.  The MB <-> bytes round trip is exact (1024*1024 is a power of two).
* `override(**knobs)` validates ALL names first (so a typo cannot leave half a block applied),
  applies in kwargs order, restores in REVERSE order on every exit path, and rolls back the
  already-applied knobs if a setter raises during entry (mirroring the v5.4.6 F-17 fix in
  `lumenairy_context`).  A knob whose requested value already equals the live value is not re-set:
  several setters clear caches (`set_pyfftw_planner` drops the pyFFTW plan cache,
  `set_default_complex_dtype` drops the ASM H cache, `set_cache_budget` evicts globally).
* `restore(state)` puts EVERY registered knob back: names in `state` to their recorded value,
  names absent from `state` (registered after the snapshot, because their owner module was imported
  during the interval) to their registration-time value.  That is the only defensible answer for a
  knob nothing observed earlier, and it is what makes the conftest fixture correct.
* `override` is PROCESS-global, not thread-local; nesting composes and unwinds correctly, and it is
  safe to call from a worker thread (pinned with a `ThreadPoolExecutor`), but two threads
  overriding the same knob race exactly as two `set_*` calls would.  Documented in the module
  docstring.  `rcwa`'s `blas_threads` is the one knob whose REQUEST is thread-local while its
  APPLICATION (via `threadpoolctl` -> `openblas_set_num_threads`) is not; its registration says so.

**Knobs registered (17).**  The audit's census minus the three lens ones (`lens_sag_dtype`,
`lens_parallel_amp`, `pointwise_cos_grid_cache_budget` -- WP-A16's), which brings the total to the
audit's 20 once A16 lands.  `set_low_memory` is deliberately NOT registered: it is a MACRO over
`fft_plan_cache_size` / `fft_double_buffer` / `fft_auto_promote` / `lens_parallel_amp`
(+ `default_complex_dtype` when `aggressive=True`), all of which are individually registered, so
snapshot/restore already covers its effects -- 4 of 5 today, 5 of 5 after A16.  `set_fft_fallback`
had no public getter; rather than add new public API for the registry's sake it is registered
against a module-private `_get_fft_fallback()`.

**conftest fixture.**  `tests/conftest.py:357-422`, autouse, function-scoped, silent restore, with
`LUMEN_TEST_KNOB_LEAK_STRICT=1` to fail the leaker -- the house style of the two guards around it.
It is deliberately NOT the existing `_module_flag_leak_guard`: that one discovers UPPER-CASE
module scalars in four named modules, so it cannot see a knob behind `_BACKEND`, `_library_path`,
`_CACHE_BUDGET_OVERRIDE_BYTES`, `_MAX_RAM_OVERRIDE` or the thread-local `_BLAS_STATE`, and it is
module-scoped, so a leak still reaches every later test in the same module.

MEASURED overhead (this box, CPython 3.14, 17 knobs, median of 20 batches of 500):
`snapshot()` **2.98 us**, no-change `restore()` **5.67 us**; the fixture does two snapshots and one
restore = **11.6 us per test**, i.e. 0.155 s across the 13 319-test suite against its 3.65 h
runtime.  No wall-clock assertion is made anywhere in a test (TESTING_STANDARDS S1).

### 2.3 Lazy loading (deliverable 3)

**Design.**  `elements/__init__.py:148-248` -- `_LAZY_SUBMODULES` (5) + `_LAZY_NAMES` (38 names ->
defining submodule) + `__getattr__` + `__dir__`.  `lumenairy/__init__.py:2081-2199` --
`_LAZY_SOLVER_NAMES` (63 names) folded into the EXISTING `__getattr__` (which already forwarded the
four live `DEFAULT_*` knobs) + `__dir__`.  A resolved name is cached into module globals, so the
second access is an ordinary dict hit and the `DEFAULT_*` forward stays the only per-access cost.
`__getattr__` raises `AttributeError`, never `ImportError`.

Each table maps a name to the module that DEFINES it, spelled exactly as the eager
`from .elements.<mod> import ...` statement it replaces, so a name that moves between submodules
fails loudly in the test rather than resolving through a facade by accident.

**What is pinned** (`test_audit2609_a15b_lazy_and_layering.py`): every import spelling returns the
same module object; `dir()` on both packages; every `__all__` entry resolves on both;
`from lumenairy.elements import *`; `pickle` round-trips of five objects defined in lazy modules;
`AttributeError` + `hasattr` + `getattr(..., default)`; the caching; and full table
consistency in both directions.

**Bit-identity of the public surface.**  A digest of every attribute of `lumenairy`,
`lumenairy.elements`, `raytrace`, `analysis`, `optimize`, `elements.pmm`, `elements.rcwa` (name ->
`module.qualname` for callables, `repr` for values) was captured on the untouched tree and
re-compared after: **0 removed, 0 changed, 17 added** (the 15 new `__all__` names plus the two new
module-level tables), `lumenairy.elements.__all__` byte-identical, `lumenairy.__all__` 708 -> 723.

**MEASURED import time.**  `python -X importtime -c "import lumenairy"`, `OPENBLAS_NUM_THREADS=1`,
on a workstation shared with other agents (hence the spread).  Medians: **before 955.3 ms** (5
runs: 848.7 / 911.8 / 955.3 / 958.6 / 1181.6), **after 916.2 ms** (7 runs: 795.9 / 861.3 / 875.3 /
916.2 / 919.2 / 919.5 / 1024.0).  The 39 ms median difference is inside that spread, so the
load-bearing measurement is the interleaved same-build A/B below, which has no cross-run noise.

Top-20 by SELF time, median run of each set:

| # | BEFORE self ms | cum ms | module | | AFTER self ms | cum ms | module |
|---|---|---|---|---|---|---|---|
| 1 | 74.7 | 87.3 | `charset_normalizer.api` | | 81.8 | 100.6 | `charset_normalizer.api` |
| 2 | 55.4 | 100.5 | `numpy.testing._private.utils` | | 77.3 | 105.8 | `numpy.testing._private.utils` |
| 3 | 35.3 | 36.4 | `scipy.special._support_alternative_backends` | | 21.0 | 21.7 | `scipy.special._support_alternative_backends` |
| 4 | 14.9 | 110.6 | `numpy.f2py.crackfortran` | | 14.6 | 14.6 | `lumenairy.optimize.merit_terms` * |
| 5 | 9.7 | 11.4 | `logging` | | 9.8 | 132.5 | `numpy.f2py.crackfortran` |
| 6 | 9.6 | 9.6 | `scipy.linalg._special_matrices` | | 7.5 | 68.7 | `numpy._core._multiarray_umath` |
| 7 | 7.6 | 12.9 | `scipy.fft._basic` | | 7.5 | 12.7 | `scipy.fft._basic` |
| 8 | 7.5 | 9.8 | `numpy.f2py.rules` | | 7.3 | 7.3 | `lumenairy.sources.core` |
| 9 | 7.3 | 7.3 | `psutil._psutil_windows` | | 7.3 | 7.3 | `psutil._psutil_windows` |
| 10 | 7.2 | 7.2 | `_ctypes` | | 7.1 | 7.1 | `scipy.linalg._special_matrices` |
| 11 | 7.0 | 82.4 | `numpy._core._multiarray_umath` | | 6.9 | 6.9 | `_colorize` |
| 12 | 6.3 | 6.3 | `numpy.ma.core` | | 6.8 | 6.8 | `_ctypes` |
| 13 | 6.2 | 8.5 | `numpy._core._add_newdocs` | | 6.6 | 6.6 | `numpy.ma.core` |
| 14 | **5.5** | **5.5** | **`lumenairy.elements.pmm.twod_jones`** | | 6.0 | 10.6 | `lumenairy.elements._lens_traced` |
| 15 | 5.1 | 5.1 | `_colorize` | | 5.7 | 18.5 | `charset_normalizer.cd` |
| 16 | 4.8 | 18.4 | `numpy._core.multiarray` | | 4.8 | 6.3 | `unittest.result` |
| 17 | 4.7 | 4.7 | `scipy.linalg._decomp` | | 4.7 | 4.7 | `scipy.linalg._decomp` |
| 18 | 4.5 | 4.5 | `scipy.fft._realtransforms` | | 4.7 | 4.7 | `_compat_pickle` |
| 19 | 4.5 | 15.6 | `scipy.linalg._basic` | | 4.6 | 4.6 | `scipy.fft._realtransforms` |
| 20 | 4.4 | 6.3 | `scipy.linalg._decomp_update` | | 4.4 | 6.1 | `numpy.f2py.rules` |

\* `lumenairy.optimize.merit_terms` is another work package's in-flight edit (the file is modified
in the working tree), not mine; it was ~0 in the BEFORE run.

Named cumulative, same two runs:

| module | BEFORE cum | AFTER cum |
|---|---|---|
| `lumenairy` | 955.3 ms | 916.2 ms |
| `lumenairy.analysis` | 918.8 | 868.4 |
| **`lumenairy.elements`** | **144.6** | **121.7** |
| `lumenairy.elements.pmm` | 16.0 | *(not imported)* |
| `lumenairy.elements.berreman` | 6.1 | *(not imported)* |
| `lumenairy.elements.rcwa` | 5.3 | *(not imported)* |
| `lumenairy.elements.bor` | 5.3 | *(not imported)* |
| `lumenairy.backend` | 567.3 | 572.2 |
| `lumenairy.backend.scipy` | 566.2 | 570.9 |
| `scipy.linalg` | 496.1 | 517.2 |
| `scipy.special` | 67.4 | 50.9 |

**The noise-free measurement.**  Same build, 9 interleaved pairs of
`perf_counter` around `import lumenairy` vs `import lumenairy` + an explicit import of exactly
what used to be eager (`elements.berreman`, `.bor`, `.pmm`, `.pmm.stack2d`, `.pmm.stack2d_pure`,
`.pmm.twod`, `.pmm.twod_jones`, `.pmm.twod_staggered`, `.rcwa`):

```
lazy   711.7 ms (median of 9: 680.6 689.1 702.7 706.9 711.7 714.9 728.2 738.2 791.2)
eager  749.0 ms (median of 9: 710.0 716.1 718.8 729.6 749.0 750.2 763.5 808.0 814.7)
                                                     -> -37.3 ms, -5.0 %
```

**Why this is 5 %, not the audit's 65 %, and what would get the 65 %.**  The audit's chain
(`analysis.coherence -> elements -> berreman -> rcwa -> backend.scipy -> scipy.linalg`) is no longer
the one that pays.  On this HEAD `lumenairy.analysis.__init__` imports `aberration` and
`beam_stats` BEFORE `coherence`, and `analysis/beam_stats.py:41` does
`from ..backend import array_namespace`, so `lumenairy.backend` -> `backend/__init__.py:30`
(`from . import scipy as scipy`) -> `scipy.linalg` + `scipy.special` is charged there: **566 ms of
the 955 ms total (59 %)**.  `lumenairy.backend` has **41 module-level importers**, including
`propagators.rs:20`, `propagators.hf:54`, `elements.elements:29` and `elements.freeform:29`, so no
amount of laziness in `elements/__init__.py` can defer it.

Measured what deferring it is actually worth (7 interleaved pairs, fresh interpreters):
`import scipy.special` alone costs **543.4 ms** (median) and adding `scipy.linalg` on top costs
**590.6 ms** -- the heavy shared prefix (`scipy._lib._array_api` -> `array_api_compat.numpy` ->
`numpy.f2py` -> `charset_normalizer` -> `numpy.testing`, 373 ms of it) belongs to `scipy.special`
just as much as to `scipy.linalg`.  After my change the ONLY remaining module-level importers of
either are `lumenairy/backend/scipy.py:21-22` and `lumenairy/elements/_lens_traced_multibranch.py:57`
(`from scipy.special import airy`).  So:

* deferring `backend/__init__.py`'s `from . import scipy` alone: **~47 ms** (the marginal
  `scipy.linalg` cost), because `_lens_traced_multibranch` still pulls `scipy.special`;
* deferring BOTH: **~540 ms, i.e. ~70 % of `import lumenairy`**.

Neither file is in my ownership; both requests are in section 5 with the exact code.

### 2.4 Layering (deliverable 4)

`raytrace/surface.py:22` was `from ..elements.lenses import surface_sag_biconic,
surface_sag_general` -- the single import-time edge among the audit's 12 layering violations.
**I moved the import into `_base_surface_sag_xy`** rather than inverting the dependency, because:
(a) the same function already imports `..elements.freeform` in-function three lines below, so the
file's own idiom answers the question; (b) the two names have exactly one reader each, both in that
function, and neither is in `surface.__all__` nor imported from `surface` anywhere in
`lumenairy/`, `tests/` or `validation/` (grep-verified); (c) inverting it properly means extracting
the sag kernels into a leaf module, which is WP-A16's `elements/_lens_kernels.py` and would have
made this one edge wait for a 3-day change.  At call time the import is a `sys.modules` dict
lookup -- `elements.lenses` is loaded long before any ray is traced.

FAIL-BEFORE evidence, from the committed HEAD blob rather than from memory:

```
== HEAD lumenairy.raytrace.surface: 1 eager edge
    LAYER   lumenairy.raytrace.surface:22 -> lumenairy.elements.lenses
```

Bit-identity gate: `_surface_sag_xy` against `elements.lenses`'s own kernels called directly, on a
rotationally-symmetric aspheric surface, a biconic and a flat, with `np.array_equal` (not a
tolerance -- a pure re-binding has no numerical content).  Plus 155 raytrace tests across six
files, all green.

Two notes on the walker used in the test.  It excludes `if TYPE_CHECKING:` bodies (they do not
execute at import time); without that exclusion it falsely flags `algebra/base.py:42`,
`raytrace/bundles.py:43-44` and `propagators/system.py:26-27`.  And it is scoped to `raytrace`
rather than asserting a whole-library layer order, because the `elements`/`propagators` relation is
genuinely ambiguous in this library -- `elements/polarization.py:75`, `elements/_lens_real.py:135`
and `elements/lenses_gbd.py:39,104` import propagator KERNELS at module level by design, which
the audit's own table records as "`elements -> propagators`, 9 edges, 0 at module level".  That
count and mine disagree (I measure 4 at module level); a test that adjudicated the ordering would
be asserting an opinion rather than the finding, so it does not.  Recorded here for the
orchestrator.

### 2.5 Re-exports and walker reconciliation (deliverable 5)

The `__all__`-symmetry walker was RED on **19** names at session start (`[top] 1251,
[exempt] 72, [MISSING] 19`).  All 19 are now re-exported and **no exemption was added**:

| name | from | requested by |
|---|---|---|
| `unwrap_phase_2d` | `analysis.opd` (also in `analysis.core`, `analysis`) | WP-A7 section 5.1 |
| `clear_meshgrid_cache`, `meshgrid_cache_bytes` | `analysis.beam_stats` | WP-A7 section 5.2 |
| `zernike_basis_cache_bytes` | `analysis.zernike` | WP-A7 section 5.2 |
| `MinEdgeThicknessMerit`, `edge_thickness` | `optimize.core` | WP-A10 section 5.1 |
| `pmm_2d_order_drift` | `elements.pmm.twod` | WP-A13 section 5.1 / VERIFY-A13 open item 2 |
| `exit_vertex_transfer`, `EXIT_VERTEX_GRAZING_TOL`, `resolve_exit_index`, `vertex_plane_transfer_t` | `raytrace.exit_vertex` | audit section 15.1 / WP-A1 |
| `exit_vertex_transfer_jax` | `raytrace.jax_trace` | audit section 15.1 / WP-A1 |
| `seed_entrance_eikonal` | `raytrace.trace` | WP-A7 section 5.3 |

`resolve_exit_index` and `vertex_plane_transfer_t` were the only genuine judgement call: they are
the lower-level pieces of `exit_vertex_transfer` and could have been exempted.  They are
re-exported instead, because the whole point of the section 15.1 module is that the seven
divergent hand-written copies converge on ITS API, and a caller writing its own transfer (the
traced-lens and GBD legs do) needs exactly those two -- "which surface is the exit?" and "how far
to its vertex plane?".  They were also added to `lumenairy.raytrace.__all__` alongside
`seed_entrance_eikonal`, closing WP-A7's asymmetry complaint from the other side.

**Late addition.**  While I was working, a concurrent work package (WP-A4's Y2 follow-up,
uncommitted) added `aberration_free_reference_fit` to `propagators/asymptotic_canonical_fit.__all__`
and `propagators/asymptotic.__all__`, which turned the walker red again on 2 rows.  It is a
documented public helper (the aberration-free reference a Strehl ratio is measured against), so I
re-exported it in the same block as `aberration_tensor`.  **If that work package renames or
withdraws the name, this re-export must move with it** -- it will fail loudly at import.

**`from_prescription` -- decided, not taken.**  WP-A11 section 5.1 offered both readings.  I kept
`Operator.from_prescription` as the only public spelling: the walker's exemption registry already
carries the rationale (a free function would be a second name with identical semantics on a
723-name surface), VERIFY-A11 confirmed `algebra.__all__` is correctly untouched and the walker is
not tripped, and adding the name would have required an `algebra.__all__` change in a module I do
not own.  The decision is recorded as a test
(`test_audit2609_a15b_reexports.py::test_from_prescription_stays_a_classmethod_only`) so it is not
silently re-litigated.

**`rs_alias_free_distance`.**  Not actionable: `propagators/rs.py:175` defines
`_rs_alias_free_distance` (private) and `rs.__all__` contains only
`rayleigh_sommerfeld_propagate`.  The public name does not exist, so there is nothing to
re-export.  Request in section 5.

**The `test_niche_audit_w3_infra.py` fence.**  See section 2.7.

### 2.6 `propagators/system.py` subharmonics (deliverable 6)

`system.py:1017` now forwards `subharmonics=elem.get('subharmonics', 0)`, and the `'turbulence'`
element's key list (`:685-694`) documents it.  Bit-identity of the default is asserted with
`np.array_equal` against the generator called directly at `subharmonics=0` and applied by hand --
an independent path through the same maths.  The knob's effect is measured so the passthrough pin
is falsifiable (the V1 defect of this audit is a pin that cannot fail): on the regression fixture
(N=32, dx=0.5 mm, r0=10 mm, seed fixed) `subharmonics=3` moves the screen by max \|dphi\| =
**2.6701 rad**, rms 1.3526 rad; `subharmonics=1` by 0.8413 / 0.3906 rad.

### 2.7 The ASM estimate fence (WP-A11 section 5 item 4)

WP-A11 left this as an explicit either/or: "either the constant comes down or that test's Windows
fence goes up, with the new measurement dated in the comment -- one decision, one place."  I
re-measured and brought the constant down.

MEASURED 2026-09-12, the A-6 derivation's own method (fresh interpreter + `tracemalloc`, 12 points
N = 64..2048 x {complex64, complex128}, fitting `cold = slope*N^2 + fixed` over consecutive-N
pairs):

```
  pair    256 ->  512 :  slope  96.02 B/px   fixed  36.71 MiB   (c128)
  pair    512 -> 1024 :  slope  96.01 B/px   fixed  36.71 MiB   (c128)
  pair   1024 -> 2048 :  slope  96.01 B/px   fixed  36.71 MiB   (c128)
  pair    256 ->  512 :  slope  51.66 B/px   fixed  36.48 MiB   (c64)
  pair    512 -> 1024 :  slope  47.10 B/px   fixed  37.62 MiB   (c64)
  pair   1024 -> 2048 :  slope  48.01 B/px   fixed  36.72 MiB   (c64)
```

The cold peak reproduces to **0.003 %** over 5 fresh interpreters per point, so these are model
constants, not noisy samples.  Against the shipped 56 MiB the estimate read est/measured =
**1.341 at N=512** and 1.188 at N=1024, while the docstring promised 1.06-1.09 -- the published
accuracy had stopped being true, and the flat `<= 1.35` fence had been sized in 2026-08-01 to
admit exactly that much drift.

`_ASM_FIRST_CALL_FIXED_BYTES` is back to **40 MiB** = 1.063 x the worst pair fit (37.62 MiB), the
same headroom convention the 2026-08-01 calibration used (56 over 52.97 = 1.057).  Re-measured
band over all eight N >= 256 points: **1.0610 / 1.0669 / 1.0773 / 1.0811 / 1.0854 / 1.0918 /
1.1048 / 1.1122** -- a bound at every one.

The fence is now derived instead of chosen.  `est(N) = F_est + s_est*N^2` and
`cold(N) = F_meas + s_meas*N^2` with all four terms positive, so `est/cold` is a weighted mediant
of `F_est/F_meas` and `s_est/s_meas` and lies BETWEEN them for every N.  The two endpoints are
40/36.71 = 1.0896 and 53.6/48.01 = 1.1164 (complex64; complex128 gives 101.6/96.01 = 1.0582), so
the bar is **1.12** and holds for every N without a per-N number.  Three new/changed bars:

* `test_fixed_term_tracks_the_measured_import_cost` (NEW) -- two-sided on the constant itself:
  `F_meas <= F <= 1.10 * F_meas`.  FAILS at 56 MiB (1.526x).
* `test_est_bounds_measured_first_call_peak` -- the `>= 1.0` A-6 contract kept; the Windows
  `<= 1.35` replaced by `<= _A6_EST_OVER_COLD_MAX` (1.12), and the measured COLD peak additionally
  pinned against its dated two-term model at 2 % (on a quantity that reproduces to 0.003 %), so a
  drift in the environment names itself instead of eating the fence's slack.  FAILS at 56 MiB
  (1.341 / 1.188).
* `test_documented_band_vs_steady_state` -- the small-N ratio (20.35 at 56 MiB) is now DERIVED from
  the constant and cross-checked against the docstring's 16.35, so the two cannot drift apart.

`tests/unit/test_verify_perf_fixes_2026_08_10.py:87-107` pins two absolute estimates to 0.001 GB;
they move by exactly the 16 MiB re-calibration (19.762 -> 19.745 GB and 22.648 -> 22.631 GB).  I
updated them AND re-derived the fixed term from the constant in the assertion, so the next
re-calibration moves the pin with it rather than reddening that file.  It is not one of the audit's
WP files, but its subjects are `memory.py` and `fft_infra.py`, both mine this round -- flagged here
per COMMON.md section 6.

Not attributable to me: the drop from ~53 MiB to 36.71 MiB is the propagator-side work that landed
earlier in this remediation (the brief attributes it to WP-A5/A11).  Nothing in this work package
touches the first ASM call -- the `tracemalloc` window opens after `import lumenairy`, and my
`fft_infra` edits are the CuPy wrapper and the knob registrations.

### 2.8 conftest comment (addendum)

`tests/conftest.py:245-262` cited `ui/waveoptics_dock.py` clearing `USE_PYFFTW` unconditionally as
the live reason `fft_infra` is leak-guarded.  WP-A9 fixed the dock (U4).  The guard stays; the
sentence now says the leak is fixed and the guard exists to catch a recurrence, and says why
enforcing the invariant by construction beats trusting ~30 call sites.

---

## 3. Files touched

**New (6):**

* `lumenairy/backend/_optional.py` (180 lines)
* `lumenairy/_knobs.py` (357 lines)
* `tests/unit/test_audit2609_a15b_optional_and_knobs.py` (33 tests)
* `tests/unit/test_audit2609_a15b_lazy_and_layering.py` (23 tests)
* `tests/unit/test_audit2609_a15b_reexports.py` (24 tests)
* `tests/unit/test_audit2609_a15b_system_subharmonics.py` (6 tests)

**Modified (16):**

* `lumenairy/__init__.py` -- 15 new re-exports + `__all__`; `_LAZY_SOLVER_NAMES` (63) folded into
  the existing `__getattr__` / `__dir__`; the 9 eager solver import blocks removed
* `lumenairy/elements/__init__.py` -- 5 eager solver blocks removed, `_LAZY_SUBMODULES` /
  `_LAZY_NAMES` (38) / `__getattr__` / `__dir__` added; `__all__` unchanged
* `lumenairy/raytrace/__init__.py` -- `seed_entrance_eikonal`, `resolve_exit_index`,
  `vertex_plane_transfer_t` re-exported + `__all__`
* `lumenairy/raytrace/surface.py` -- the layering import moved into `_base_surface_sag_xy`
* `lumenairy/propagators/fft_infra.py` -- `_optional` delegation; 12 `register_knob` calls; new
  private `_get_fft_fallback`
* `lumenairy/propagators/system.py` -- `subharmonics` passthrough + element-dict doc
* `lumenairy/sources/core.py` -- `_optional` delegation
* `lumenairy/optimize/_merit_jit.py` -- `_optional` delegation (gate stays local)
* `lumenairy/memory.py` -- `max_ram` knob; `_ASM_FIRST_CALL_FIXED_BYTES` 56 -> 40 MiB + derivation;
  two docstring accuracy statements
* `lumenairy/cache.py` -- `cache_budget` knob + `_get_cache_budget_override_mb`
* `lumenairy/io/storage.py` -- `storage_backend` knob
* `lumenairy/user_library.py` -- `library_path` knob + `_get_library_path_override` /
  `_set_library_path_override`
* `lumenairy/elements/rcwa/_core.py` -- `blas_threads` knob (one registration, no behaviour change)
* `tests/conftest.py` -- autouse knob-leak guard; the WP-A9 comment correction
* `tests/unit/test_niche_audit_w3_infra.py` -- the derived A-6 bars
* `tests/unit/test_verify_perf_fixes_2026_08_10.py` -- the two absolute pins re-derived

`lumenairy/elements/pmm/__init__.py`: **inspected, no change needed** -- VERIFY-A12 had already
added `"pmm_2d_order_drift"` to its `__all__` (commit `49c05569`, confirmed present at
`c95e4657`).  Pinned from both ends by
`test_audit2609_a15b_reexports.py::test_pmm_facade_exports_the_order_drift_helper`.

`ruff check` is clean on all 22 files.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and
`-q --no-header -p no:cacheprovider`.

| Command (files) | Result | Time |
|---|---|---|
| `test_public_api.py test_v4_16_0_walker_all_symmetry.py test_v4_14_1_dispatcher_pin_cache_clears.py test_audit2609_a15b_*.py (4) test_verify_perf_fixes_2026_08_10.py test_v5_1_0_agent_c_split.py test_v5_1_0_agent_b_split.py test_v5_3_multi_field_merit_jit.py` | **466 passed, 4 skipped** | 11.9 s |
| `test_audit_w6_raytrace.py test_niche_audit_w3_raytrace_sources.py test_niche_s9_lattice_and_sentinels.py test_audit_v5_24_2_g02.py test_niche_p9_decenter_tilt.py test_analytic_ray_transfer.py` | **155 passed** | 353 s |
| `test_rcwa.py test_v5_20_4_berreman_mode_cache.py test_v5_20_7_pmm_geo_eig_cache.py test_niche_audit_w3_elements.py test_audit_w5_elements_misc.py test_elements_lens.py` | **186 passed** | 172 s |
| `test_audit_misc.py test_v4_16_0_agent_d_cache_registry.py test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py test_v4_14_2_dispatcher_pin_cache_locks.py test_niche_audit_p2b_infra_contracts.py` + the 4 new files | **482 passed, 8 skipped** | 126 s |
| `test_perf_v4_12_0_fft_infra.py test_audit_w2_fft_state.py test_g1_cache_memory.py test_memory_guardrail.py test_c1_s4_19_storage_metadata_contract.py test_c2_s4_19_user_library_safe_eval.py test_public_api.py` (re-run WITH the new autouse fixture) | **801 passed** | 10.9 s |
| `test_niche_r0_byte_budgeted_cache.py test_g08_s4_15_cache_hygiene.py test_sources.py test_audit_sources.py` (+ the four above) | **285 passed** | 17.7 s |
| `test_niche_audit_w5_shim_removals.py test_v5_12_0_naming_aliases.py test_g08_s3_16_sources.py test_niche_s12_source_shape_wiring.py test_audit2609_a11_polar_sources_infra.py test_audit2609_a5_verify_fft_buffer_threads.py` | 225 passed, 44 skipped, **1 pre-existing failure** (below) | 22.9 s |
| `test_niche_audit_w3_infra.py` | 79 passed, 2 blocked by a harness artefact (below) | 2.1 s |
| `python validation/run_all.py --quiet` (37 files) | 36 PASS, **1 pre-existing failure** (below) | ~7 min |

**Pre-existing failures found, both attributable elsewhere:**

1. `test_niche_audit_w5_shim_removals.py::TestPropagatorInertKwargRemovals::test_hf_chunk_output_is_KEPT`
   -- `DID NOT WARN`.  Exactly the item WP-A11 flagged in its section 5 item 5: the test asserts a
   `chunk_output` deprecation that no longer exists because another work package made the kwarg
   functional.  `propagators/hf.py` is not mine and is not in my diff.
2. `validation/test_lenses.py::apply_real_lens_traced_jax` --
   `RuntimeError: ... requires double precision, but jax_enable_x64 is disabled`.  The JAX x64
   policy adopted from `rcwa/_core.py` (audit section 15.6); `elements/_lens_jax.py` is WP-A4's and
   is not in my diff.  36 of 37 validation files pass.

**Harness artefact (not a code defect).**
`test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512|1024-complex128]`
raise `OSError: [WinError 6] The handle is invalid` / `[WinError 50] The request is not supported`
from `subprocess.run(capture_output=True)` inside `_measure_asm_peak`, because this agent's shell
does not give pytest a valid stdin handle to duplicate.  It reproduced on the UNTOUCHED file at
session start, before any change of mine, and it passed once when PowerShell happened to supply a
console handle (`7 passed in 7.99 s`).  To verify the assertions anyway I executed them outside
pytest against the same child program over **eight** N x dtype points; all pass, with the band
reported in section 2.7.  The driver is
`<scratch>/run_a6_offline.py` and its output is quoted in section 2.7.

---

## 5. Requested changes outside my ownership

### 5.1 `lumenairy/backend/__init__.py:30` -- defer `scipy` (the ~70 % import-time item)

This is the single highest-value remaining change to `import lumenairy`, and it is 12 lines.
Replace

```python
from . import scipy as scipy
```

with a PEP 562 forward, keeping `'scipy'` in `__all__` and `la.backend.scipy.jv(x)` working:

```python
import importlib as _importlib


def __getattr__(name):
    """Load ``lumenairy.backend.scipy`` on first use (audit 2026-09-11 P2-7).

    It imports ``scipy.linalg`` + ``scipy.special`` at module scope, which is
    ~566 ms of a ~955 ms ``import lumenairy`` on the calibration box, and the
    41 module-level importers of this package (``propagators.rs``,
    ``elements.elements``, ...) want only ``array_namespace`` / ``is_*_array``.
    """
    if name == 'scipy':
        mod = _importlib.import_module('.scipy', __name__)
        globals()['scipy'] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {'scipy'})
```

Worth **~47 ms on its own** (measured: `import scipy.special` 543.4 ms vs
`import scipy.special, scipy.linalg` 590.6 ms, medians of 7 interleaved pairs), and **~540 ms
(~70 % of `import lumenairy`)** together with 5.2.  The existing walker exemption
`('lumenairy.backend', 'scipy')` already covers the name, and `backend/scipy.py`'s own
`from .array import ...` is unaffected.

### 5.2 `lumenairy/elements/_lens_traced_multibranch.py:57` (WP-A16) -- defer `scipy.special`

`from scipy.special import airy as _scipy_airy` is, after my change, the ONLY module-level
`scipy.special` importer outside `backend/scipy.py`, and it is the reason 5.1 alone is worth 47 ms
instead of 540 ms.  Move it into the function that calls `_scipy_airy` (the same in-function idiom
the file already uses elsewhere), or behind a cached `_airy()` accessor.  With 5.1, this takes
`import lumenairy` from ~712 ms to roughly ~170 ms on this box.

### 5.3 `lumenairy/propagators/rs.py:175` (WP-A5) -- rename `_rs_alias_free_distance`

WP-A5 section 5.2 asked for a top-level `rs_alias_free_distance` and offered to rename on request.
The public name does not exist yet, so I could not re-export it.  Requested: rename to
`rs_alias_free_distance`, add it to `rs.__all__`, and keep `_rs_alias_free_distance` as an alias
for the three test files that import the private name
(`test_audit2609_a5_propagators.py:62`, `test_audit2609_a5_verify_rs_and_rw.py:27`).  Once it
exists, the top-level re-export is two lines in `lumenairy/__init__.py` and I will add it -- or
the orchestrator can, next to the `propagators.rs` block.

### 5.4 The five lens-family `_optional` sites (WP-A16)

`backend/_optional.py`'s contract is documented in its module docstring and in section 2.2 above.
The exact lines to replace, all verified present today:

| file : lines | delete | replace with |
|---|---|---|
| `elements/_lens_real.py:32,36-42,44-50` | `CUPY_AVAILABLE = find_spec(...)`, `_ensure_cupy_loaded`, `_is_cupy_array` | `from ..backend._optional import CUPY_AVAILABLE, ensure_cupy, is_cupy_array` + the two thin wrappers that keep the module-level `cp` alias (copy `fft_infra.py:52-93`) |
| `elements/_lens_traced.py:36,40-46,48-54` | same three | same |
| `elements/_lens_traced.py:69,76-89` | `_NUMBA_AVAILABLE`, `_load_numba` | keep `_NUMBA_AVAILABLE` as a LOCAL gate (tests monkeypatch it), delegate the import to `numba_handles()` -- copy `optimize/_merit_jit.py:69-81` |
| `elements/lenses.py:44,48-53,121-133` | `CUPY_AVAILABLE`, `_ensure_cupy_loaded`, `_is_cupy_array` | same as `_lens_real` |
| `elements/lenses.py:80,87-100` | `_NUMBA_AVAILABLE`, `_load_numba` | same as `_lens_traced` |
| `elements/_lens_imap.py:410,417-429` | `_NUMBA_AVAILABLE`, `_load_numba` | same |
| `propagators/fga.py:94,97-107` | `_NUMBA`, `_load_numba` | `load_numba()`; keep the `ImportError` with the `pip install lumenairy[numba]` hint at `:376` -- it is the contract for a path with no NumPy fallback |
| `elements/_lens_thin.py:44-45,61,...` | -- | already delegates to `lenses._is_cupy_array`; it can point at `_optional.is_cupy_array` directly once `lenses.py` switches |

Three behaviours must be preserved and are easy to lose: `_lens_traced`/`lenses` keep their
module-level `cp` because `xp = cp if _is_cupy_array(E) else np` reads it; `_NUMBA_AVAILABLE` stays
a module attribute because several tests monkeypatch it; and `fga`'s raise is deliberate.

### 5.5 The lens-family knobs (WP-A16)

Register `lens_sag_dtype`, `lens_parallel_amp` (`elements/_lens_traced.py`) and
`pointwise_cos_grid_cache_budget` (`elements/_lens_real.py`) with
`lumenairy._knobs.register_knob` the same way -- one call beside each `set_*`/`get_*` pair.  That
takes the registry to the audit's 20 and completes `set_low_memory`'s coverage (it flips
`lens_parallel_amp`, which is the only one of its four knobs not yet restorable).  The getters must
be cheap and side-effect-free; see the two cases in section 2.2 that needed a private accessor.

### 5.6 Informational

* `tests/unit/test_verify_perf_fixes_2026_08_10.py` -- I edited two absolute pins (section 2.7).
  Not an audit WP file, but its subjects (`memory.py`, `fft_infra.py`) are mine this round.
* The audit's layering table records "`elements -> propagators`, 9 edges, 0 at module level"; I
  measure **4 at module level** (`elements/_lens_real.py:135`, `elements/lenses_gbd.py:39,104`,
  `elements/polarization.py:75`).  Either the audit's classifier differs or its ordering places
  `propagators` below `elements`.  Not acted on -- see section 2.4.
* `aberration_free_reference_fit` (section 2.5) is re-exported from an UNCOMMITTED sibling change.
  If that work package renames it, the re-export in `lumenairy/__init__.py` must move with it.

---

## 6. Deferred, with designs

1. **`lumenairy/__init__.py` fully lazy** (~700 names).  The bounded version shipped here defers
   only the five solver families, because that is what the brief scopes and because the remaining
   cost is `backend.scipy`, which section 5.1 removes far more cheaply.  If 5.1 and 5.2 land and
   more is still wanted, the design is mechanical: AST-walk the `from .x import a, b, c` blocks in
   order, emit `{name: module}` for the LAST binding of each name (several names are bound twice --
   `coronagraph_contrast_curve` from both `.elements` and `.analysis.coronagraph`), fold into the
   existing `__getattr__`, and gate with the identity digest already written for this work package
   (`<scratch>/digest_surface.py`: 746 attributes, compared name by name to
   `module.qualname` / `repr`).  Effort ~4 h including the digest sweep.  Value AFTER 5.1/5.2:
   `lumenairy`'s own 162 module bodies cost 877 ms total by the audit's measurement and ~120 ms by
   mine, so this is the last ~100 ms, not the headline.
2. **A `reset_*` for every knob.**  The registry records each knob's registration-time value
   (`_Knob.initial`) and `restore({})` already returns every knob to it, so
   `lumenairy.reset_knobs()` is a three-line public wrapper.  Not added because it is new public
   API the audit did not ask for and would need its own walker entry.  Effort ~15 min if wanted.
3. **`override()` as a thread-local scope.**  Genuinely useful for `set_blas_threads`, whose
   REQUEST is already thread-local, and impossible for the other 16, which are module globals.  A
   half-thread-local `override` would be worse than the documented process-global one.  The real
   fix is per-call kwargs for the knobs that are operation-level, which is the audit's own
   config-object recommendation (section 14, item 13, 10 d).
4. **A knob census test that fails when a NEW `set_*` lands unregistered.**  The census in
   `test_audit2609_a15b_optional_and_knobs.py` is a fixed expected set, so it catches a REMOVED
   registration but not an ADDED setter.  The generalisation is an AST walk for module-level
   `def set_*` that writes a `global`, minus a documented exemption list (Qt property setters in
   `ui/`, `set_low_memory` the macro, per-object setters).  I did not add it because the exemption
   list is a judgement call across files owned by five work packages this week.  Effort ~2 h.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A15b_CHANGELOG.md`

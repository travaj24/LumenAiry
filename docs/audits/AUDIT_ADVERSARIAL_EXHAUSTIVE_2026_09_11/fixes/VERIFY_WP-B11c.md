# VERIFY-WP-B11c -- independent adversarial re-verification of the Wave-5.3 structural refactors

Branch under test: `refactor/wp-b11c-structure`, HEAD `5268041e`, base `96cb2096`.
Verification branch: `verify/wp-b11c` in the worktree `C:/tmp/lum_vhyg1`.
I did not write WP-B11c.  Nothing in `lumenairy/` was edited by this verification.

**This is a REFACTOR verification.**  The contract is two things and only two
things: (1) BIT IDENTITY of every public answer between `96cb2096` and the
branch, and (2) preserved semantics of every monkeypatch target and every
switch the move touched.  Everything below is re-measured on fixtures the
engineer did not use, with instruments written for this verification that share
no code with `validation/probe_wp_b11c/`.

---

## Verdict table

| # | Claim (as the WP-B11c report states it) | Verdict | My numbers |
|---|---|---|---|
| 1 | The 223-line BLAS-thread cap moved from `rcwa/_core.py` to `rcwa/_blas.py` | **CONFIRMED** | the removed `_core.py:150-372` block at the base is byte-for-byte `_blas.py:36-258` on the branch (`diff` exit 0, 223 lines); 14 of 14 cap names byte-identical by `ast.get_source_segment`; `_core` 4967 -> 4764 lines |
| 2 | `rcwa/_blas.py` is a leaf: stdlib + `lumenairy._knobs` | **CONFIRMED, one wording correction** | AST + `__import__` recorder on both trees: `_blas -> {lumenairy._knobs}` and nothing else; `_knobs` itself imports no `lumenairy` module. The report's graph table row says "`rcwa._blas ->` (nothing in the package)", which is loose -- `_knobs` IS in the package. Section 1.1's prose is correct |
| 3 | `_core` re-exports the seven `__all__` names BY IDENTITY | **CONFIRMED** | probe keys `A01`-`A05`: all 95 `__all__` names resolve on `_core`; the seven are the same objects on `_core`, `lumenairy.elements.rcwa` and `lumenairy`; ordered `__all__` and its set bit-identical base vs branch |
| 4 | The cap functions' globals are `_blas.__dict__`, so the STATE is deliberately not re-exported and a stale `setattr(_core, ...)` raises `AttributeError` | **CONFIRMED and strengthened** | re-measured `__globals__ is vars(_blas)` for all 8 state readers; and re-stated in the shape a test actually uses -- `monkeypatch.setattr(_core, <name>, ...)` with the default `raising=True` refuses for all 7 state names, and binds nothing (my `test_a_stale_monkeypatch_on_core_raises_rather_than_binding_a_shadow`, 7 ids, 7 red on the base) |
| 5 | Four monkeypatching test files re-pointed, with proofs of patch; four M6 arms were vacuous on a box with `threadpoolctl` | **CONFIRMED** | `threadpoolctl` IS installed on both builds here, so the four "no warning" arms were indeed vacuous before; `_assert_patch_site_is_live()` plus the `probes` counter close it. Functionally re-measured end to end: a counter substituted at `_blas._get_blas_controller` is reached by `rcwa_efficiency_1d`, `rcwa_efficiency_2d`, `RCWAStack.solve_vs_wavelength`, `PMMStack.solve_vs_wavelength` and `PMM2DStackHybrid.solve_vs_wavelength` |
| 6 | Sag builders + CuPy/numba/numexpr plumbing moved `lenses.py` (746 -> 368) -> `_lens_kernels` | **CONFIRMED, one number corrected** | the line count is 746 -> **367** -> **346**, not 746 -> 368 -> 338 (defect D10). 19 of 19 moved lens names: 16 byte-identical source segments, 3 code-identical with only a docstring rewrite (`_ensure_cupy_loaded`, `_is_cupy_array`, and `CUPY_AVAILABLE` which is an `ImportFrom` at both ends). **Zero executable-code changes** across all 33 moved names, and **zero unlisted relocations** -- no silent passenger |
| 7 | `docs/lens_configuration.md`'s PEP 562 plan was WRONG; a read-only module `__getattr__` carries reads only | **CONFIRMED, re-derived from scratch** | `validation/probe_verify_b11c/probe_pep562_counterfactual.py` rebuilds the three mechanisms on synthetic modules with no lumenairy import: read-only `__getattr__` -> `kernel_saw_the_patch=false`, `shadow_left_in_facade_dict=true`; `ModuleType` subclass -> `true` / `false`; plain by-value -> lazy slot lost (`null`), patch not seen, shadow left |
| 8 | `lenses` installs a `ModuleType` subclass forwarding eight names on read/write/delete/`__dir__` | **CONFIRMED and extended** | all eight exercised for read, WRITE and DELETE, and for the full monkeypatch save/set/undo cycle with `name not in vars(lenses)` after each (16 ids, all 16 red on the base). Also re-checked what a module type can break: `importlib.reload` keeps the facade and the live forward, `pickle` round-trips the moved callables by identity, `inspect.getsource(lenses)` works, `dir(lenses)` is de-duplicated and sorted, and the `AttributeError` text for a missing name is **byte-identical** to CPython's own |
| 9 | The last four `lenses <-> lenses_maslov` names moved; the lens family's module-level 2-cycle count is ZERO; both ratchets restated as equalities | **CONFIRMED, with a scope note** | two independent instruments (my own AST walk, my own `__import__` hook) on both trees: base 2 family 2-cycles -> branch **0**. Every row of the report's before/after table reproduces exactly. **Scope note:** one module-level 2-cycle involving the package ROOT survives and is unchanged by this package -- `lenses_maslov`'s `from .. import raytrace` against `lumenairy/__init__.py`'s import of `lenses_maslov`. It is outside the family definition both ratchets use, so the claim is true as stated; it is recorded because "the lens family reaches zero import cycles" reads broader than it is |
| 10 | Bit identity 43/43, 40/40, 48/48 archive-to-archive on both builds | **CONFIRMED on my own fixtures, larger** | **167/167 on Windows py3.14** (55 RCWA/PMM + 112 lens) and **161/161 on the WSL venv py3.12** (55 + 106; the 6 missing are the CuPy-device keys WSL has no device for). Zero differing keys, zero only-base, zero only-branch, archive-to-archive, child processes, `lumenairy.__file__` anchored per arm |
| 11 | Fail-before: 15 failed / 1 passed on the base | **CONFIRMED exactly** | reproduced on an isolated `git archive 96cb2096`: `15 failed, 1 passed in 3.79s`, the same 15 ids, the same single green (`test_the_module_getattr_still_raises_for_a_name_nobody_defines`) |
| 12 | 28 CHANGELOG citations re-anchored BY CONTENT | **CONFIRMED exactly** | my own content walker (`probe_citations.py`) reads each citation's line text at `96cb2096` and at HEAD: 282 citations checked, **exactly 28 moved, 28/28 with identical line text, 0 content-differing** -- including the one that changed FILE (`rcwa/_core.py:249` -> `rcwa/_blas.py:135`). `scripts/check_source_line_citations.py` independently reads drift 0, 107/107 |
| 13 | Whole-grid surface body NOT done; three things it would move recorded | **CONFIRMED (as a deferral)** | the three reasons are visible in the source: the numexpr decision is taken on `E.size` in `_lens_real.py:7257` / `:7485` / `:8073`, `_ensure_full_grids` is reached only on the whole-grid path, and the Fresnel promotion sits at a different pipeline step. Correctly deferred; it is not a bit-identical refactor |
| 14 | The one red (`test_s10_vector_normalisation...`) is not this package's | **CONFIRMED, and now BOUNDED** | see section 4 |
| 15 | "CuPy. Not installed on this box" (report's "what could not be measured") | **REFUTED** | CuPy 14.0.1 IS installed on the Windows py3.14 build with a working device; `lumenairy.backend.CUPY_AVAILABLE` is `True`. The GPU arms were therefore measurable and I measured them -- `_is_cupy_array` on a device array, `_ensure_cupy_loaded`, and `surface_sag_general` / `surface_sag_biconic` dispatching to `xp = cp` on the device -- **bit-identical base vs branch** (`probe_lens.py` keys `D01`-`D08`). The claim is wrong; the conclusion it supported is still right |
| 16 | "a refactor whose contract is that nothing observable moves" | **RESTATED** | one observable surface the report does not record: `__module__` moved for all 16 callables this package moved (10 lens + 6 BLAS), and `type(lenses).__name__` moved from `module` to `_LensesFacade`. **21 of 30** introspection keys differ, identically on both builds. Inherent to any module split (part a did the same) and harmless in-tree, but it changes what `pickle` writes, what a traceback prints and what Sphinx resolves |

---

## 1. What I measured, and with what

Worktrees: `C:/tmp/lum_vhyg1` (branch `verify/wp-b11c` off `refactor/wp-b11c-structure`)
and `C:/tmp/lum_vhyg1_pre` (detached at `96cb2096`).  **Neither was used as a
measurement arm.**  Both arms of every bit-identity comparison are
`git archive` extractions into the scratch directory, so a concurrent edit in
any checkout on this shared box cannot move a number underneath the
measurement.  Every python run carried
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line; every pytest run carried `--capture=sys -p no:randomly`.

Instruments, all new for this verification (`validation/probe_verify_b11c/`):

| file | what it does |
|---|---|
| `vlib.py` | bind-and-refuse + SHA-256 digest + a `record()` that folds the value's TYPE, its bytes, and every warning in EMISSION ORDER (the package harness sorts warnings, so a reordering is invisible to it) |
| `run_bitid.py` | runs one probe twice in child processes, one per archive, and diffs the digest maps key by key |
| `probe_rcwa.py` | 55 keys: the cap's public surface, RCWA 1-D (TE/TM/oblique/metallic-Li/laurent/stabilize/deep), 1-D Jones both spellings, a wavelength sweep, `rcwa_extrapolate`, 2-D crossed / conical / circular-truncation / analytic shapes, 2-D Jones on a full anisotropic tensor cell at normal and conical incidence, `RCWAStack` solve + serial and threaded sweeps, the same arithmetic inside `rcwa_blas_threads(1)`, `(4)`, `(None)`, `_blas_threads_quiet(2)` and a process-global `set_blas_threads(2)` and uncapped again, and the PMM consumers (1-D TE/TM/oblique, 1-D Jones, `PMMStack` serial + threaded, `PMM2DStackHybrid` sweep + Jones, `pmm_efficiency_2d_cell` capped and uncapped) |
| `probe_lens.py` | 112 keys (106 on WSL): 26 sag keys over every branch of both builders including the refusals, F-ordered input, float32 and the odd-power guard; the numba and numexpr gates flipped THROUGH the facade; the facade's read/write/delete/`dir`/`hasattr` surface; the two fitting helpers over a 20-point index grid and 21 normaliser cases including degenerate ones; **the real CuPy device**; and the consumers -- thin (plain and decentred), spherical, aspheric, cylindrical, real (singlet / three-surface cemented doublet / slant+fresnel / aspheric / complex64 / absorption / seidel), traced, traced-multibranch, Maslov x3, Maslov-vector x4 normalisations, GBD, FGA, `Surface` sag, the grid-vs-aperture helpers with their warnings, and the DOE leg (`create_diffractive_lens` + `apply_doe_phase_traced` at two orders) |
| `probe_surface.py` | 30 keys of INTROSPECTION metadata (`__module__`, `__qualname__`, type) for every moved name, plus the `AttributeError` text for three missing names |
| `probe_graph.py` | the import graph twice -- an `ast.NodeVisitor` that refuses to descend into any function/class body, and an `__import__` hook that keeps only module-body calls (`f_locals is f_globals`); `--first M` imports a module BEFORE `lumenairy` |
| `probe_moved_source.py` | for every moved name, the exact source segment at both ends, plus a docstring-stripped `ast.dump` so a prose edit is told apart from a code edit; plus the adversarial half, every relocated name the report does NOT list |
| `probe_pep562_counterfactual.py` | the three forwarding mechanisms rebuilt on synthetic modules, with no lumenairy import |
| `probe_citations.py` | every CHANGELOG `path:line` citation's LINE TEXT at the base vs at HEAD |
| `probe_s10_ulp.py` | the hand-off red's quantity over a family of 12 fixtures |

Hash JSONs for both builds are committed beside them
(`*_win_*.json`, `*_wsl_*.json`, `*_compare.json`).

---

## 2. Task A -- bit identity on my own fixtures

| arm | build | keys | identical | differing | only-base | only-branch |
|---|---|---|---|---|---|---|
| RCWA + PMM | Windows py3.14 | 55 | **55** | 0 | 0 | 0 |
| lens family | Windows py3.14 | 112 | **112** | 0 | 0 | 0 |
| RCWA + PMM | WSL py3.12 | 55 | **55** | 0 | 0 | 0 |
| lens family | WSL py3.12 | 106 | **106** | 0 | 0 | 0 |

**328 key comparisons, 328 identical.**  The six Windows-only lens keys are
exactly the CuPy-device block (`probe_lens.py` keys `D03`-`D08`); WSL has no CuPy, both arms there
agree on that and produce the same 106 keys.

Three things about this measurement are worth stating because they are what
makes it adversarial rather than confirmatory:

* **Archive-to-archive.**  The probe file is passed by ABSOLUTE path from the
  verification worktree and is not in either archive, so both arms run
  byte-identical probe source against different libraries.  Each arm prints
  `lumenairy.__file__` and `SystemExit`s if it is not under its own archive.
* **The record folds more than the answer.**  Exception type AND message,
  every warning's category and text IN ORDER, and the returned object's type
  name.  A refactor that moved a warning's `stacklevel`, reordered two
  warnings, or returned a `list` where a `tuple` was returned is caught.
* **The cap is exercised set AND unset.**  `probe_rcwa.py` keys `D01`-`D09`
  run the identical
  1-D solve uncapped, inside `rcwa_blas_threads(1)`, `(4)` and `(None)`,
  inside `_blas_threads_quiet(2)`, under a process-global `set_blas_threads(2)`,
  and uncapped again -- the only arithmetic a BLAS-thread cap can reach.

### The verbatim check, which is stronger than any hash

A refactor whose contract is bit identity has a cheaper and stricter gate
available than a numeric probe: the moved definitions' SOURCE must be
byte-identical.  `probe_moved_source.py` over the 9 files this package could
have touched:

* **33 moved names**: 30 `VERBATIM` (byte-identical source segment),
  3 `CODE-IDENTICAL-DOCSTRING-CHANGED`, **0 `CODE-CHANGED`**.
* **0 unlisted relocations** -- no name changed module that the report does not
  name.
* **0 in-place text changes** -- no name stayed in its file with different text.

The whole 223-line BLAS block is a single `diff` with exit 0:
`git show 96cb2096:lumenairy/elements/rcwa/_core.py | sed -n '150,372p'` equals
`sed -n '36,258p' lumenairy/elements/rcwa/_blas.py`.

---

## 3. Tasks B, C, D -- monkeypatch semantics, the graph, the cap state

### B. Every moved or forwarded name

The report's eight live-forwarded names (`cp`, `_ne`, `NUMEXPR_AVAILABLE`,
`_NUMBA_AVAILABLE`, `_numba`, `_njit`, `_prange`, `_NUMBA_KERNELS`) each get
TWO new ids in `tests/unit/test_verify_b11c_structure.py`: read + WRITE +
DELETE through the facade with `name not in vars(lenses)` asserted after each,
and the full `monkeypatch` save/set/undo cycle.  All 16 ids are red on the
base.

`_fit_normaliser` and `_multi_indices_total_degree` are re-exported by value and
are pure functions; they are covered by identity (`lenses.X is
_lens_kernels.X is lenses_maslov.X`) and by 24 bit-identity probe keys
(`probe_lens.py` `C01`-`C04`: a 20-combination `(n_vars, order)` index grid,
21 normaliser cases including all-zero, constant, a pair straddling zero at
1e-9 and a NaN, each over three `pad` values, and the `lenses_maslov`
spelling's identity with the `lenses` one).

**A durability asymmetry I found (defect D3 below).**  The eight live names are
forwarded; eight OTHER moved names are re-exported into `lenses` with `X as X`
*and* are read at call time out of the leaf's globals:
`CUPY_AVAILABLE`, `_is_cupy_array`, `_ensure_cupy_loaded`, `_load_numba`,
`_get_aspheric_sag_accum_numba`, `_ensure_numexpr_loaded`, and part a's
`_collect_semi_diameters` and `_warn_if_aperture_exceeds_grid`.  At `96cb2096`
a `monkeypatch.setattr(lenses, '_is_cupy_array', fake)` DID change
`surface_sag_general`'s behaviour, because the kernel lived in `lenses`.  On
the branch the same line succeeds, binds a shadow in `lenses.__dict__` and is
read by nothing -- which is precisely the silent no-op the report's section 1.3
says the design set out to avoid, arriving with the opposite failure mode from
the one it fenced.  No test in the suite does this today; nothing is red.

### The `ModuleType` subclass, stress-tested

| probe | result |
|---|---|
| `importlib.reload(lenses)` | same object, class still `_LensesFacade`, no shadow bound for any of the eight, forward still live afterwards |
| `pickle.dumps/loads` of the moved callables | round-trips BY IDENTITY; the payload now names `_lens_kernels` |
| `inspect.getsource(lenses)` / `getsourcefile` | work; 16 099 chars, `.../elements/lenses.py` |
| `dir(lenses)` | sorted, de-duplicated, lists all eight; base vs branch **bit-identical** as a probe key |
| the `__all__`-symmetry walker, the PEP-562 forwarding walker, the shell-vs-canonical walker | green (section 6) |
| `AttributeError` message for a missing name | **byte-identical** to CPython's own on three probes (`surface_sag_generl`, `zzz_no_such_name`, `_NUMBA_AVAILABL`) |

### C. The import graph, reproduced and attacked

Both of my instruments, on both trees:

| | `96cb2096` | branch |
|---|---|---|
| module-level 2-cycles, lens family | `(_lens_real, lenses)`, `(lenses, lenses_maslov)` | **none** |
| module-level 2-cycles, whole package | those two + `(lumenairy, lenses_maslov)` | `(lumenairy, lenses_maslov)` |
| `_lens_real ->` | `_lens_kernels`, `lens_config`, `lenses` | `_lens_kernels`, `lens_config` |
| `lenses_maslov ->` | `_lens_kernels`, `_lens_real`, `lens_config`, `lenses` | `_lens_kernels`, `_lens_real`, `lens_config` |
| `_lens_kernels ->` | (nothing in the family) | (nothing in the family; `backend._optional` outside it) |
| `rcwa._core ->` (package-internal + the `_knobs` leaf) | `_geometry`, `lumenairy._knobs` | `_blas`, `_geometry` |
| `rcwa._blas ->` | (does not exist) | `lumenairy._knobs`, nothing else |

Every row of the report's table reproduces.  The AST walk and the recorder
agree on every row, on both trees.

**Attacks, all on both trees:** the six modules imported FIRST in a fresh
interpreter (`rcwa._blas`, `rcwa._core`, `_lens_kernels`, `lenses_maslov`,
`_lens_real`, `lenses`).  All twelve runs exit 0 with the same cycle counts
(base 3, branch 1) and the same edge sets; no `ImportError`, no partially
initialised module survives, and importing the new leaf first is indistinguishable
from importing it last.

**One wording correction on the instrument.**  The report says the rcwa rows
were measured "on `import lumenairy`".  `rcwa` is PEP-562 LAZY: a bare
`import lumenairy` never loads `rcwa._core`, and my first recorder run returned
no rcwa edges at all.  The package's recorder reaches them only because it
imports with `fromlist=("*",)`, which triggers the lazy `__getattr__`.  The
substance is right; the sentence is not what it describes.  I reproduced the
rows by importing `lumenairy.elements.rcwa._core` explicitly.

### D. The cap state

* `set_blas_threads` / `rcwa_blas_threads` still act: probe keys `A06`-`A09`
  round-trip `None, 1, 4, 0, -3, 7, None` through the knob and read the
  `max(1, int(n))` floor, enter and restore both context managers, and record
  the `ValueError` for a non-numeric request -- all bit-identical.
* `threadpoolctl` is present on BOTH builds here (Windows py3.14 and the WSL
  venv), so the "absent on WSL" premise in my brief does not hold on this box;
  the inert-cap arms are reached only through the substituted
  `_threadpoolctl_available`, which is what the re-pointed M6 block now does.
* `setattr(_core, '<state name>', ...)` raises as claimed, for all seven names,
  in the `monkeypatch` shape (7 new ids, 7 red on the base).
* **Every `_core` consumer of the cap still reaches the leaf's state**, measured
  functionally rather than by inspection: a counter substituted at
  `_blas._get_blas_controller` is reached by `rcwa_efficiency_1d`,
  `rcwa_efficiency_2d`, `RCWAStack.solve_vs_wavelength(max_workers=2)`,
  `PMMStack.solve_vs_wavelength(max_workers=2)` and
  `PMM2DStackHybrid.solve_vs_wavelength`.  (`pmm_efficiency_1d` is NOT capped
  at either commit -- the 1-D PMM entry points carry no `_with_blas_limit` --
  so it is not in the list.)

---

## 4. Task E -- the s10 ULP red, bounded

The report hands this red on, arguing (a) the quantity is bit-identical
archive-to-archive, so the red is not this package's, and (b) re-deriving the
bar belongs to
`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py`'s owner.  My brief
says to bound it.  Both halves check out, and the bound is worse than the
report suggests.

### It is not this package's -- confirmed on twelve fixtures, not one

`validation/probe_verify_b11c/probe_s10_ulp.py` recomputes the test's own
quantity -- `P_x/P_y` under `normalize_output` in `'none'`, `'power'` and
`'peak'`, and the ULP delta of each against `'none'` -- over a FAMILY of twelve
fixtures: the test's exact one, then neighbours that vary only the amplitude
pair (4 values), the grid size (80 / 96 / 112 / 128), the quadrature node count
(40 / 48 / 56), the polynomial order and the waist.  Run in child processes
bound to each archive.

**Every one of the twelve rows is identical between `96cb2096` and the branch**
-- not close, the same numbers:

| fixture | power ULP | peak ULP | | fixture | power ULP | peak ULP |
|---|---|---|---|---|---|---|
| exact test fixture | 3 | 1 | | N = 80 | 2 | 0 |
| amp 0.9 / 0.5 | 0 | 1 | | n_v2 = 40 | 3 | 2 |
| amp 0.6 / 0.8 | 1 | 1 | | n_v2 = 56 | 2 | 0 |
| amp 0.99 / 0.14 | 2 | 2 | | poly_order 5 | 0 | 0 |
| N = 112 | 0 | 0 | | waist 0.50 mm | 1 | 1 |
| N = 128 | 0 | 1 | | waist 0.60 mm | 1 | 0 |

And the test itself passes, alone, on all four combinations:

| | base tree | branch |
|---|---|---|
| Windows py3.14 | `1 passed in 22.77s` | `1 passed in 24.45s` |
| WSL py3.12 | `1 passed in 14.07s` | `1 passed in 14.35s` |

### The bar IS sample-scoped -- the bound

* The exact fixture reads **3 ULP for `'power'`, 1 for `'peak'`** on BOTH trees
  and on this build today.  The docstring records **"0 ULP for 'power', 1 ULP
  for 'peak'" (WP-A4-VERIFY, 2026-09-13)**.  So the docstring's recorded
  measurement is already stale against the build the bar is running on -- the
  WP's re-measurement of 3 and 1 is right, and it is not a branch effect.
* Across the family the delta reaches **3 for `'power'` and 2 for `'peak'`**
  against a **4-ULP** bar.  That is **one ULP of headroom at the top of a
  twelve-point sample** -- exactly the S4 shape `docs/TESTING_STANDARDS.md`
  names: the pass/fail boundary sits inside the spread of the quantity.
* The bar's derivation in the comment is the reason it is too tight.  It reasons
  "one joint scale `s` makes the ratio `(a s)^2 / (b s)^2`, which is exact up to
  the rounding of the two products: bar 4 ULP".  But `P_x` and `P_y` are not two
  products -- each is a **9216-term reduction** of `|E|**2` (N**2 = 96**2), and
  rescaling every element by `s` perturbs each term before the sum.  A bound
  derived from the reduction rather than from "two products" is the missing
  piece; 4 ULP is a plausible number for the wrong model.
* A run-order effect is still untraced.  The report says importing `jax` and
  enabling `jax_enable_x64` first (what the two files ahead of it in the
  ten-file slice do, without restoring) does not move the reading.  Re-measured
  here over the whole twelve-fixture family with `--jax-first`:
  **all twelve rows identical to the no-jax branch arm**, exact fixture still
  3 and 1, `r0` still `1.7777776632250892`.  Confirmed.  What moves it is
  somewhere else in that run's history, and the family measurement says only
  that a one-ULP nudge is enough to cross the bar -- which is what a bar with
  one ULP of headroom means.

**Verdict.** The report's conclusion is CONFIRMED and its argument is
strengthened from "the hash did not move" to "the quantity does not move on
twelve fixtures".  The bar is BOUNDED as sample-scoped: it survives the twelve
fixtures I could build with one ULP to spare, which is not a gap.  Re-deriving
it -- from the reduction length, with the envelope measured across the run
orders that move it -- remains the owner's work, and this section is the
measurement they need to start from.

---

## 5. Task F -- durability of `tests/unit/test_audit2609_b11c_structure.py`

The file is sound: 16 ids, every one a structural fact, no float / mode count /
timing anywhere, and the fail-before is exactly as reported.  Four durability
gaps, all LOW, all fenced by new ids in
`tests/unit/test_verify_b11c_structure.py` rather than by an edit to the WP's
file:

1. **The patch-site inventory is form-specific.**
   `test_the_four_monkeypatching_files_patch_the_definition_site` recognises
   `monkeypatch.setattr(_core, 'NAME', ...)` and `_core.NAME = ...`, and only
   when `_core` arrived via `from ...elements.rcwa import _core`.  It does not
   see the STRING target form
   (`monkeypatch.setattr('lumenairy.elements.rcwa._core._BLAS_CONTROLLER', ...)`),
   `import lumenairy.elements.rcwa._core as C`, or anything under
   `tests/integration`.  My `test_no_test_file_patches_the_cap_state_on_core_in_any_form`
   covers all three, walks `tests/` whole, and is careful about the trap that
   bit my own first cut: three of the four re-pointed files bind `rcwa._blas`
   to a local name *spelled* `_core`, so a name-based heuristic gives four
   false positives.
2. **The restated leaf-closure walk mis-resolves a package `__init__`.**
   `test_audit2609_b11_hygiene.py::test_nothing_the_leaf_can_reach_imports_the_elements_family`
   computes a relative import's base as `mod.split('.')[:-1]`, which is right
   for a module and wrong for a package: inside `lumenairy/backend/__init__.py`
   a `from ._optional import x` would resolve to `lumenairy._optional`, which
   has no file, and the walk `continue`s -- silently dropping that subtree.
   Today's closure is `{lumenairy.backend._optional}`, a plain module, so the
   answer is right and the property holds; the walk that produced it would not
   survive a leaf import through any subpackage.  My
   `test_the_leaf_closure_reaches_no_elements_module_with_packages_resolved`
   re-derives it with a resolver that handles both, and FAILS on an unresolvable
   node instead of skipping it.
3. **Four of the sixteen ids `pytest.skip` when `validation/probe_wp_b11c/import_graph.py`
   is absent** (an installed wheel).  `docs/TESTING_STANDARDS.md` restatement 4
   is explicit that a skip on an environment check silently removes tests on
   exactly the runners that matter.  My
   `test_the_rcwa_core_reads_the_blas_leaf_and_the_leaf_has_no_back_edge` and
   `test_the_lens_family_carries_no_module_level_two_cycle_by_a_second_walk`
   restate the same claims from the SOURCE, so they still gate there.
4. **`test_the_four_moved_names_resolve_from_both_ends_and_are_identical`
   includes `NUMEXPR_AVAILABLE`,** whose `is` comparison between two `bool`
   singletons cannot distinguish a live forward from a stale snapshot unless
   the two values actually differ.  It is not wrong; it is weaker than the other
   three arms of the same test.  The live property is covered elsewhere
   (`test_the_numexpr_gate_is_live_through_the_facade_too`), so no new id.

My file: **39 ids, 39 green on the branch, 34 red on the base**, with the five
base-green ids declared rather than counted as evidence (they are guards:
`_knobs` was already a leaf; `dir(lenses)` already listed the eight names when
they were real globals; `check_grid_vs_apertures` moved in part a; nobody
patched a by-value re-export at the base either; the leaf's closure was already
clean).

---

## 6. Task G -- runs

All runs carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
on the command line, `--capture=sys -p no:randomly`, and `PYTHONPATH` pinned to
this worktree.  "win" is Windows py3.14 (numpy 2.4.4), "wsl" the WSL venv
py3.12.  The box carried other agents' work throughout, so the durations are
upper bounds.

| run | build | tail |
|---|---|---|
| `test_rcwa*` + `test_niche*rcwa*` + `test_lens*` + `test_*maslov*` + `test_pmm*` + `test_audit2609_b11*` + the four re-pointed monkeypatch files + my DECISION file (34 files) | win | **`1098 passed, 1 skipped`** in 25:14 -- the skip is `test_niche_audit_m4_m5_m6_rcwa.py:428`, "threadpoolctl installed: the cap is effective here" |
| census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget` + the `__all__`-symmetry walker (27 files) | win | **`624 passed, 11 skipped`** in 2:57 -- `test_public_api::test_installed_metadata_version_matches_source_version` GREEN here, and `_BLAS_CONTROLLER_LOCK` still found on the cache-lock exemption list after the move |
| every test that actually opens `CHANGELOG.md` (14 files, V12 / V17 / V18 / doc-consistency / the two stamps) | win | **`117 passed, 6 skipped`** in 1:30 -- the six are the clean "nothing to verify" skips on the 5.47.0 block |
| `b11c_structure` + `b11_hygiene` + my DECISION file + the four monkeypatch files + `test_rcwa` + the two a16 arch files (10 files) | wsl | **`369 passed, 1 skipped`** in 3:26 -- same skip, so `threadpoolctl` is present on the WSL venv too |
| my DECISION file alone | win | **`39 passed`** in 3.8 s |
| my DECISION file on an isolated `git archive 96cb2096` | win (base) | **`34 failed, 5 passed`** in 2.6 s |
| `test_audit2609_b11c_structure.py` on the same isolated base | win (base) | **`15 failed, 1 passed`** in 3.8 s -- the WP's claimed fail-before, id for id |
| `ruff check lumenairy/ tests/ validation/probe_verify_b11c/` | wsl | **All checks passed!** |
| `scripts/check_source_line_citations.py` (V18) | win | `summary: ok=107  drift=0  total=107`, exit 0 |
| `scripts/record_history_fingerprints.py --check` | win | `OK: every history document matches its module.`, exit 0 |
| `.test_durations` | win | 16 207 entries, all 16 `b11c` ids present -- as claimed |

`-X importtime` on both archives, importing `lumenairy` then
`lumenairy.elements.rcwa._core`: the only structural difference is the new
`lumenairy.elements.rcwa._blas` line on the branch.  `lumenairy.elements.lenses`
is logged TWICE on BOTH trees, so that is not evidence of the cycle either way
-- which is exactly the report's point that `-X importtime` cannot see the edge
a cycle is made of.


---

## 7. Defects

| id | severity | what | reproducer |
|---|---|---|---|
| **D1** | LOW (report factual error, no code impact) | The report's "What could not be measured" says "**CuPy.** Not installed on this box".  CuPy 14.0.1 IS installed on the Windows py3.14 build and has a working device; `lumenairy.backend.CUPY_AVAILABLE` is `True`.  The GPU arms of `_is_cupy_array` / `_ensure_cupy_loaded` / the sag builders' `xp = cp` dispatch were measurable and are now measured -- bit-identical base vs branch on a real device. | `python -c "import lumenairy.backend as b, cupy; print(b.CUPY_AVAILABLE, cupy.zeros(3).sum())"`; `probe_lens.py` keys `D01`-`D08` in `lens_win_*.json` |
| **D2** | LOW (observable surface, inherent) | `__module__` moved for all 16 callables this package moved -- 10 `lumenairy.elements.lenses` -> `..._lens_kernels` and 6 `rcwa._core` -> `rcwa._blas` -- and `type(lenses).__name__` moved from `module` to `_LensesFacade`.  **21 of 30** introspection keys differ, identically on both builds.  This changes the `pickle` payload path, the module a traceback names and what Sphinx resolves.  Harmless in-tree and unavoidable in any module split (part a did the same), but the report records the `dir()` surface as "an observable surface change in a refactor whose contract is that nothing observable moves" and does not record this one. | `validation/probe_verify_b11c/surface_{win,wsl}_compare.json` |
| **D3** | LOW (latent, durability) | Eight moved lens names are re-exported into `lenses` BY VALUE and read at call time out of the leaf's globals, so a stale `monkeypatch.setattr(lenses, '_is_cupy_array', fake)` SUCCEEDS and reaches nothing -- the silent no-op the BLAS half was deliberately arranged to make loud.  At `96cb2096` that same line DID change the kernel's behaviour. | fenced by `test_no_test_file_substitutes_a_by_value_reexport_on_the_lenses_facade` and documented by `test_the_names_lenses_re_exports_by_value_are_read_out_of_the_leaf` |
| **D4** | LOW (durability) | The WP's patch-site inventory misses the string-target form, the plain-`import`-as form and `tests/integration`. | fenced by `test_no_test_file_patches_the_cap_state_on_core_in_any_form` |
| **D5** | LOW (durability) | The restated leaf-closure walk mis-resolves relative imports inside a package `__init__` and silently drops the subtree. | fenced by `test_the_leaf_closure_reaches_no_elements_module_with_packages_resolved` |
| **D6** | INFO (wording) | The graph table row "`rcwa._blas ->` (nothing in the package)" -- it imports `lumenairy._knobs`.  Section 1.1 says it correctly. | `probe_graph.py --mode exec --first lumenairy.elements.rcwa._blas` |
| **D7** | INFO (wording) | "Measured by both instruments, on `import lumenairy`" for the rcwa rows.  `rcwa` is PEP-562 lazy; a bare `import lumenairy` loads no rcwa module.  The package's recorder reaches them only via `fromlist=("*",)`. | `graph_exec_branch.json` has no rcwa key; `graph_first_branch__core.json` does |
| **D8** | INFO (over-general prose) | Report section 2.4 describes the numba fast-path answer and the pure-NumPy answer as "equal to the fast path, bit for bit".  That is a reading of ITS coefficient set, not a property: numba's LLVM is free to contract `sag + c*h**k` into an FMA, and on `{4: 3.1e2, 6: -8.4e6, 8: 1.7e11}` at 64 x 64 (measured here 2026-09-15) the two arms differ in the last bits.  The probe key itself is fine -- it hashes the same thing on both trees. | `tests/unit/test_verify_b11c_structure.py::test_the_numba_gate_flip_through_the_facade_changes_which_arm_runs` |
| **D9** | INFO (stale in-code cross-reference) | `lumenairy/elements/_lens_traced.py:2639` cites "``rcwa/_core.py``'s ``_threadpoolctl_available``"; that function now lives in `rcwa/_blas.py`.  `lumenairy/_knobs.py:58` similarly cites `elements/rcwa/_core.py::set_blas_threads`.  Both are prose in comments, outside the V18 citation walker's reach. | `grep -rn "_core.*_threadpoolctl_available" lumenairy/` |
| **D10** | INFO (wrong number) | Report section 2.1 says `lenses.py` "drops **746 -> 368** lines at this item (338 after item 3)".  Measured from the commits themselves: **746 -> 367 -> 346**.  The `_core.py` figures (4967 -> 4764) and the 223-line block size are exact. | `for R in 96cb2096 5b462ff2 e05790dc; do git show $R:lumenairy/elements/lenses.py \| wc -l; done` |

No P0, no P1, no regression.  Nothing I found changes a number the library
returns.

---

## 8. Ship recommendation

**SHIP.**  The contract this package set itself -- bit identity of every public
answer, and preserved semantics of every monkeypatch target -- holds under
independent re-measurement on 328 key comparisons over fixtures the engineer did
not use, on two builds, archive-to-archive; and it holds at a level a numeric
probe cannot reach, because the moved code is byte-for-byte the same source with
zero executable-code changes and zero unlisted relocations.  The two design
decisions that carry real risk -- refusing a PEP 562 read-only forward, and
refusing to re-export the cap's state -- were both re-derived from scratch here
and are both right.

Before merge, a handful of one-line corrections worth folding into the WP's
report (none of them blocks):

* D1: strike the "CuPy not installed" sentence; the arms are measurable on this
  box and are measured.
* D6/D7/D9/D10: the `_blas ->` table row, the "on `import lumenairy`" phrasing,
  the two stale in-code cross-references to `rcwa/_core.py`, and the
  `lenses.py` line counts (367 and 346, not 368 and 338).

D3 is the one worth an owner's attention beyond wording: the lens side's stale
patches fail silently where the BLAS side's fail loudly, and the only thing
standing between that and a vacuous test is the inventory I added.

---

## 9. What I could not verify

* **The full two-lane release matrix.**  Out of scope for a package branch, as
  the WP says.  Everything below it is run.
* **A GPU-vs-CPU numerical equivalence.**  I exercised the CuPy arms and
  compared base-to-branch on the device (which is the refactor question); I did
  not compare the device answer to the host answer (which is not).
* **The `_JAX_EIG_STABLE` hazard end to end.**  Same reason the WP gives:
  demonstrating it means building the mechanism the item declines to build.  The
  argument is sound and the same shape WAS built and measured for `lenses`.
* **Whether the `test_public_api` order effect shares a cause with the glass
  one-shot pin.**  Both pass alone; I did not trace either, and neither file is
  in this diff.
* **The hand-off red's two-sided envelope across RUN ORDERS.**  I bounded it
  across FIXTURES (section 4), which is what settles whether the bar is
  sample-scoped.  Which run history pushes it over remains untraced, and
  re-deriving that bar is still
  `test_audit2609_a4_verify_maslov_asymptotic.py`'s owner's work.
* **Timings.**  This box carried up to twenty concurrent python processes from
  other agents throughout; every wall-clock number here is an upper bound and
  none of them is load-bearing.

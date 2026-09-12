# WP-A22 — final code pass (audit 2026-09-11)

Eleven items: the gate debts other work packages left behind, the last
wall-clock assertion in `tests/`, the `scipy.fft` import cost WP-A16 measured
and could not fix, and the maintenance path the WP-A17 history pin was missing.

Everything below was MEASURED on branch `audit-fixes-2026-09`, Python 3.12,
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

---

## 1. Summary

| item | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| 1. `pyproject.toml` F541 ignore | **fixed** | `pyproject.toml:428-434` (entry deleted, group comment rewritten) | `ruff check lumenairy tests scripts` | ruff itself | `ruff --isolated --select F541 lumenairy/elements/_lens_real.py`: 2 at the WP-A15a base → **0**; ignore removed, tree clean |
| 2. except census | **fixed** | `tests/unit/test_audit_except_budget.py:32-38, 51, 127-141, 149-152, 246-248` | `test_audit_except_budget.py` (4 passed) | line census over `lumenairy/**` minus `ui/` | census 51 / tree 50 / slack 1 → **census 50 / tree 50 / slack 0** |
| 3. last wall-clock assertion | **fixed** | `tests/unit/test_ci_kernel_consistency.py:79-87, 148-167, 511-513, 518-588` | `test_ci_kernel_consistency.py` | the committed census (`decisions.json`) | `assert elapsed < 20.0` → 24 probed rows == 24 cheap census rows, 0 expensive, 4 hypotheticals; wall clock 1.7 s printed |
| 4. doc-identifier exclusions | **fixed** | `scripts/check_doc_identifiers.py:196-202, 226-231` | `test_audit2609_a21_doc_identifiers.py` (5 passed) | the resolver over the four documents | denominator **593 → 597**, unresolved **0 → 0**, triaged list 51 → 47 |
| 5. docs | **fixed** | `Migration-Guide.md:1288-1290, 1441-1517`; `README.md:21-22, 3664-3671`; `CONTRIBUTING.md:63-114` | `test_audit2609_a21_doc_identifiers.py`, `test_niche_p8_capstone.py`, `test_v4_15_agent_f.py`, `test_v4_16_2_agent_d.py`, `test_v4_16_2_dispatcher_pin_doc_consistency.py`, `test_v5_2_walker_changelog_changeset.py` (211 passed, 4 skipped) | the identifier resolver + executing the snippet | snippet runs and is `np.array_equal` to the loose-keyword call |
| 6. JAX collection-order dependency | **fixed** | `tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py:44-94` | the file itself, run ALONE | the file's own 35 ids | alone: **12 failed / 23 passed → 0 failed / 35 passed** |
| 7. root `__init__` on the mypy whitelist | **not done, by design** | — | `mypy` (whitelist) | `mypy --strict` on the file | **33 errors**, unchanged; the prerequisite has not landed — list in §2.7 |
| 8a. `_lens_thin` 2-cycle | **BLOCKED** (file dirty) | patch in §5.1 | — | module-level-only AST cycle walk | **4 → 3** cycles, six entry points bit-identical — verified on an overlay copy |
| 8b. `stack2d_pure` stacklevel | **BLOCKED** (file dirty) | patch in §5.2 | — | frame probe at the raising site | shipped `stacklevel=3` reports `stack2d_pure.py:1441`; **4** reports at the user's `solve()` |
| 9. `scipy.fft` made lazy | **fixed** | `lumenairy/propagators/fft_infra.py:110-176, 2085-2093, 2331-2333, 2373-2375, 2527` | 7 FFT/ASM/propagation files (248 passed) | `-X importtime` + interleaved A/B; `sys.modules` | `import lumenairy` **633.3 → 298.1 ms (-52.9 %)**; `scipy.special` no longer loaded |
| 10. fingerprint recorder | **fixed** | new `scripts/record_history_fingerprints.py`; `docs/history/fft_infra.md:3-8`; `CONTRIBUTING.md:74-114`; `tests/unit/test_audit2609_a17_history_relocation.py:33-46, 252-261, 270-277` | new `tests/unit/test_audit2609_a22_history_fingerprint_tool.py` (11 passed) | the checker's own `ast_fingerprint` / `token_fingerprint` | drift on a temp copy re-recorded and re-verified; `fft_infra` re-recorded for item 9 |

---

## 2. Per item

### 2.1 `pyproject.toml` — the F541 TODO ignore is gone

WP-A15a parked `"lumenairy/elements/_lens_real.py" = ["F541"]` in the
per-file-ignores with a `TODO(audit-2609)` comment naming the two
placeholder-free `f"..."` continuation lines in the mirror guard.  WP-A16
dropped the prefixes.  MEASURED before deleting the ignore, so the deletion is
not taken on trust:

```
$ python -m ruff check --isolated --select F541 lumenairy/elements/_lens_real.py
All checks passed!
```

The line and its comment are deleted.  The surrounding group comment now
records that the TODO group is **empty** and that it should stay that way — a
TODO ignore is a deferred finding, not a convention.  `ruff check lumenairy`
and `ruff check lumenairy tests scripts` both pass.

### 2.2 The broad-except census comes down to 50

`_lens_imap.build_inverse_map`'s RAM-budget `from .. import memory` was the
last entry in the census's group (5), NARROWING REQUESTS — clauses "counted
because they exist, NOT endorsed".  It is now `except ImportError:`
(`_lens_imap.py:1578`).  MEASURED on this tree with the test's own census
function:

```
tree total  = 50
census total= 51            (before this change)
OVER  : {}
STALE-HIGH: {'elements/_lens_imap.py': (1, 1)}   <- allowed 1, tree has 0
```

The entry is **deleted** rather than set to 0, which is the procedure the
group comment asks for and the one WP-A21 used for `memory.py`: an implicit
allowance of zero, so the next broad except that drifts into that file fails
the gate instead of landing in free space.  Group (5) is now empty, and the
comment says why that is the state to defend.

Three stated numbers updated in the same pass so none goes stale: the group
(1) header (`34 of 51` → `35 of 50` — the 34 was itself already stale, the
group sums to 35), the scalar-bar comment (`51 across 26 files` → `50 across
25`), and the counter-pin's MEASURED line (`census 51, tree 51` → `census 50,
tree 50, slack 0`).

Worth recording: WP-A21 §5.2 asked whoever owned `elements/lens_config.py` to
decide about three `except Exception:` clauses in `_same`.  **They are gone** —
the committed file has `except (TypeError, ValueError)` / `except KeyError` and
zero broad clauses, so no census entry is needed and that request is closed.

`test_audit_except_budget.py`: 4 passed in 0.37 s.

### 2.3 `test_ci_kernel_consistency.py:492` — the last live wall-clock assertion

**What was wrong.**  `assert elapsed < 20.0` was standing in for the module
docstring's BUDGET promise: the current-arm re-take runs the four CHEAP
sections and leaves the expensive mortar / T22 sections (4.7 s and 5.3 s by
that docstring's own measurement) to the committed table.  A seconds bar
cannot express that.  It fails when this shared box is loaded — a dozen agents
run on it — and, worse, it PASSES the regression it exists to catch: a fast
enough machine can probe both expensive sections inside 20 s.

**What it is now**, per WP-A15a §6 item 3 ("assert on the NUMBER OF DECISIONS
probed"), with the number **derived from the committed census** rather than
transcribed, so regenerating the table moves the bar with it:

* the re-take's row set equals the census's cheap-row set (union over all
  twelve arms) — fewer means a section stopped being re-taken and the gate
  went partly blind there, more is the `uncensused` check from the other side;
* zero rows from the expensive prefixes (`mortar/`, `t22/`);
* the recorded number of hypothetical bars, because a hypothetical is the
  standing evidence that an unguarded site is undecidable and losing one
  silently retires that argument;
* `_RETAKE_DECISIONS = 24` as a counter-pin on the census itself, so a table
  that shrank is reported rather than silently accommodated.

MEASURED by running the four cheap probes directly:

```
elapsed 1.705 s
len(here) 24   len(hyp) 4   len(rea) 15   len(cls) 8
here prefixes {'pmm1d_interface': 6, 'sliver': 2, 'band': 4, 'branch_cut': 12}
census cheap union 24        here == census cheap union?  True
```

against 54 rows per measured census arm (the other 30 being mortar 15 + t22
15).  Both sides exact, gap 0.

**TEETH, measured rather than asserted.**  Each new bar was exercised against
a mutated copy of the probe's output: injecting one `mortar/` row fails the
expensive bar; dropping the `band` section leaves 20 of 24 rows and fails the
coverage bar; dropping one hypothetical fails the hypothetical bar.

The wall clock is now PRINTED (`arm 'WIN-unknown-t1' re-took 24 decision(s) +
4 hypothetical(s) in 1.71 s`) and asserted nowhere — TESTING_STANDARDS S1.
The module docstring's BUDGET paragraph was rewritten to say that the
restriction is enforced as a row count and why a seconds bar cannot do it.

**Sweep**: this was the last one.  `grep` over `tests/` for `assert` against
an elapsed/`perf_counter` quantity returns only numeric physics bars now.

**Residual, and it is NOT mine — see §4.**  That test still fails on this
tree, at step 1 (the RULE check), which I did not touch, and it fails
identically on the **HEAD blob** of the same file.

### 2.4 `scripts/check_doc_identifiers.py` — three exclusions retired

`LensGeometry` / `LensNumerics` / `LensResources` were on the hand-triaged
exclusion list as "ROADMAP name, since landed — resolves on its own", kept only
so the gate did not depend on WP-A16 being committed.  WP-A16 is committed
(`c7c9ebbb`), so they were deleted.  MEASURED with the script itself:

```
BEFORE   API-claiming denominator (distinct): 593     resolve: 593   DO NOT resolve: 0
AFTER    API-claiming denominator (distinct): 596     resolve: 596   DO NOT resolve: 0
```

593 → 596 exactly: three names moved from "excluded" to "resolved".  A fourth
entry, `RCWA2DPrepared.solve`, was deleted as a consequence of the
Migration-Guide edit in §2.5 (final denominator **597** after the doc
additions).  `_CURATED_CEILING` in the gate is a `<=` bar (51) and the list is
now 47, so the gate needed no edit — and it is not mine.
`test_audit2609_a21_doc_identifiers.py`: 5 passed in 7.7 s.

### 2.5 Documentation

**`Migration-Guide.md:1289` — the dead class spelling.**  The guide named both
`PreparedRCWA2D.solve` and `RCWA2DPrepared.solve` because the shipped
Wood-anomaly warning used the wrong one.  Confirmed the real spelling ships
(`twod.py:1300,1302` both say `PreparedRCWA2D.solve`; the only remaining
`RCWA2DPrepared` occurrences in the tree are in WP-A21's own test explaining
the fix).  The sentence now reads "before v5.46.0 the warning named a class
that does not exist, so grepping a captured pre-5.46 log for the real name
finds nothing" — the fact is kept, the dead identifier is not.  Its triaged
exclusion was deleted in the same change, which is what WP-A21 §5.4 asked for.

**Three late packages' notes, added to the 5.46.0 section.**

* **WP-A16, the config objects.**  Stated as NEW API with **"NO MIGRATION IS
  REQUIRED: this is purely additive"** in bold, the five entry points still
  taking every keyword with the same defaults and the same answers, the
  frozen-dataclass properties, `from_kwargs` / `to_kwargs` / `narrowed_to`, the
  refusal on mixing an object with the loose keyword it also sets, and a
  pointer to `docs/lens_configuration.md`.
* **WP-A17, the history relocation.**  Not user-facing, so it leads with
  "contributors only; no behaviour change", explains the two fingerprints, and
  carries the re-record rule from item 10 with the commands.
* **WP-A21, three corrections.**  The committed `scripts/check_doc_identifiers.py`
  gate; the PMM shared-grid **message** filter
  `-W "error:.*eps_cell is a SEGMENT grid:UserWarning"` together with the
  reason a `module=` filter cannot work at any *correct* stacklevel (the
  spelling was checked against `CI_FILTER` in
  `test_audit2609_a21_pmm_warning_filter.py:49` and the comment block at
  `twod_staggered.py:602`, after a first draft of mine got it wrong); and the
  `fast_analytic_phase` tooltip's real figure, ~7 nm rms **per mm of glass**.

**Every snippet executes.**  The one snippet added is the config-object call,
and it was run end to end on a real prescription:

```
snippet OK -> (128, 128) complex128
bit-identical to the loose-keyword form: True
round-trip: True                 # LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg
```

That is also the claim the note makes, so the snippet and the sentence are
verified by the same run.

**`README.md`.**  Two rows added to the "Start here" table — one for
`docs/lens_configuration.md`, one for `docs/history/` (with the contributor
rule) — and a short pointer block at the head of the "Real-lens accuracy
strategy" section, which is the lens section a reader of these options is
actually in.

**One mid-course correction, recorded because it is the kind of thing this
gate exists for.**  My first draft wrote the word `filterwarnings` in
backticks; the identifier resolver flagged it as an unresolved API claim
(it is a pytest ini option, not a lumenairy name).  I reworded rather than
adding an exclusion — the ceiling had room, which is exactly why taking it
would have been the wrong move.

### 2.6 `test_v4_14_0_dispatcher_pin_apply_lens.py` — a gate that depended on collection order

**Confirmed on the unmodified file**, run alone:

```
12 failed, 23 passed, 5 warnings in 2.09s
E RuntimeError: apply_real_lens_traced_jax: the JAX (differentiable) real-lens
  path requires double precision, but jax_enable_x64 is disabled -- ...
```

All twelve are the `apply_real_lens_traced_jax` / `apply_real_lens_maslov_jax`
parametrisations of the four pins.  `jax.config` is process-global; about
twenty other modules in this suite enable x64 at import, so in a full run this
file inherited it.

**Fix.**  `jax.config.update('jax_enable_x64', True)` in the same `try:` block
that decides whether JAX exists, plus a module-scoped autouse fixture that
re-asserts it at run time and RESTORES the previous value on teardown.  Both
halves earn their place:

* the import-time set is what the repo's other JAX modules do, and it is the
  earliest point;
* the fixture is needed because **imports all happen before any test runs**,
  and two modules here deliberately switch the flag OFF to exercise refusal
  paths (`test_niche_audit_w7_rcwa.py:915`, `test_v5_12_0_audit_fixes.py:113`);
* the restore is not cosmetic — leaving the flag on would export this file's
  precondition to whatever runs next, which is the defect being fixed, pointed
  the other way.

**MEASURED, alone, before → after: 12 failed / 23 passed → 0 failed / 35
passed (6.55 s).**  Cross-checked against the neighbour that toggles the flag:
`test_v4_14_0_... test_v5_12_0_...` → 40 passed, and the reverse order → 40
passed.

### 2.7 Root `lumenairy/__init__.py` on the mypy whitelist — NOT added

The prerequisite has not landed.  MEASURED,
`mypy --strict --python-version 3.12 --follow-imports silent
--ignore-missing-imports lumenairy/__init__.py`:

```
total errors: 33
  line  149: 30 x [attr-defined]
  line  799:  1 x [attr-defined]
  line 2192:  1 x [no-untyped-def]     Function is missing a type annotation
  line 2220:  1 x [no-untyped-def]     Function is missing a return type annotation
```

Identical in shape and count to what WP-A21 §5.1 measured, so nothing moved
since.  The 31 `does not explicitly export attribute` rows are, in full:

> `PreparedAnalyticLens`, `PreparedTracedLens`, `TiltedCarrier`,
> `apply_aspheric_lens`, `apply_axicon`, `apply_cylindrical_lens`,
> `apply_grin_lens`, `apply_real_lens`, `apply_real_lens_gbd`,
> `apply_real_lens_maslov`, `apply_real_lens_maslov_jax`,
> `apply_real_lens_traced`, `apply_real_lens_traced_jax`,
> `apply_real_lens_traced_multi`, `apply_real_lens_traced_multibranch`,
> `apply_real_lens_traced_segmented`, `apply_real_lens_traced_uniform`,
> `apply_spherical_lens`, `apply_thin_lens`,
> `clear_pointwise_cos_grid_cache`, `close_worker_pool`,
> `get_lens_parallel_amp`, `get_lens_sag_dtype`,
> `get_pointwise_cos_grid_cache_budget`, `lens_sag_float32_opd_error`,
> `prepare_real_lens`, `prepare_real_lens_traced`, `set_lens_parallel_amp`,
> `set_lens_sag_dtype`, `set_pointwise_cos_grid_cache_budget` (all from
> `lumenairy.elements.lenses`, at `:149`), and `trace_world` (from
> `lumenairy.raytrace`, at `:799`).

Under `--strict` a re-export must be in the defining module's `__all__` or be
spelled `from X import Y as Y`.  The two `no-untyped-def` rows are the root's
own PEP 562 pair.  So `pyproject.toml`'s `[tool.mypy] files` is unchanged and
`test_audit2609_a15a_packaging.py::test_the_mypy_whitelist_only_grows` keeps
its floor — raising a ratchet for a module that is still red would be the
defect that gate exists to catch.  Requests in §5.3 and §5.4.

`pyproject.toml`'s own comment at `:521-526` already explains this and is
still accurate, so it was left alone.

### 2.8 (item 9) `fft_infra.py` — `scipy.fft` deferred

**What was wrong.**  `fft_infra.py:112`'s module-level `import scipy.fft as
_scipy_fft` was, as WP-A16 §2.6 traced, the only reason `scipy.special` (and
its ~370 ms shared prefix) loaded at `import lumenairy`.  Confirmed on this
tree before touching anything:

```
scipy modules after import lumenairy: 85
scipy.special loaded: True   scipy.fft loaded: True   scipy.linalg loaded: False
```

**What changed.**  The pyFFTW treatment already in the same file:
`find_spec('scipy.fft')` at import for the availability answer, and a
first-use `_ensure_scipy_fft_loaded()` accessor for the module.  The four
readers (`_scipy_or_numpy_fft2`, `_scipy_or_numpy_ifft2`, `_fft2_nd`,
`_ifft2_nd`) call the accessor; `_ensure_scipy_fft_loaded` joins `__all__`
beside `_ensure_pyfftw_loaded`.

Two design points that are not the obvious spelling, both forced by
constraints outside this file:

1. **The handle is a one-slot LIST, not a rebound module global.**
   `propagation.py` must live-forward every rebound `fft_infra` global through
   PEP 562 (`_LIVE_FORWARD_NAMES`) because an import-time snapshot goes stale
   on a rebind — `cp` and `pyfftw` are both in that list, and
   `test_v5_2_walker_pep562_forwarding.py::test_v14_every_mutable_fft_infra_global_is_in_live_forward_names`
   AST-walks this file and requires it.  A `global _scipy_fft` rebind would
   therefore have needed an edit to `propagation.py`, which I do not own, and
   the walker would have been RED until it landed.  A container that is
   appended to is never rebound, so it needs no entry **and cannot acquire the
   bug the forwarding contract exists to prevent**.  The why-comment at the
   declaration says exactly this.
2. **A PEP 562 module `__getattr__` keeps `fft_infra._scipy_fft` resolving.**
   `propagators/_bluestein.py:175` reads `_fi._scipy_fft.ifft` / `.fft`
   directly (guarded by `USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE` on the line
   above).  That file is not mine and is in the sweep's scope, so the
   attribute had to keep working; reading it performs the same first-use
   import the accessor does.  Unknown names still raise `AttributeError`.

`find_spec` on a dotted name imports the parent package, so it is wrapped in
`except (ImportError, ValueError)` — a scipy that is absent, or broken
mid-install with `__spec__ is None`, answers at import rather than at the
first FFT.  MEASURED, that parent import costs **27.6 ms** with NumPy already
loaded, against the **396.0 ms** it defers.

**Structural gate (the one the brief asks to assert), MEASURED:**

```
scipy.fft in sys.modules after import lumenairy:   False
scipy.special in sys.modules:                       False
numpy.f2py: False | numpy.testing: False | charset_normalizer: False
scipy in sys.modules: True            <- the find_spec probe, 27.6 ms
SCIPY_FFT_AVAILABLE = True
after one FFT -> scipy.fft loaded: True | scipy.special loaded: True
fi._scipy_fft -> scipy.fft
bit-identical vs numpy.fft.fft2: True
```

So the claim holds with nothing left over: no other import path pulls
`scipy.special` into `import lumenairy`.

**Import time, WP-A15b's interleaved same-build method** (fresh interpreter
per sample, arms alternating pair by pair, medians of 9):

| tree | arm | median | the 9 samples (ms) |
|---|---|---|---|
| BEFORE | bare | **633.3** | 616.8 617.3 620.3 623.0 **633.3** 636.7 648.1 651.1 691.5 |
| BEFORE | eager | 628.3 | 614.6 621.7 623.3 623.8 **628.3** 635.6 652.1 668.9 669.2 |
| | | **-5.0 ms** | both arms already eager — the control reads zero, as it must |
| AFTER | bare | **298.1** | 279.6 290.3 295.5 297.2 **298.1** 301.8 308.7 309.0 310.7 |
| AFTER | eager | 622.2 | 600.4 602.4 606.2 610.9 **622.2** 629.6 636.3 644.7 673.9 |
| | | **+324.1 ms** | the cost now deferred |

Two independent numbers agreeing: plain before/after medians
**633.3 → 298.1 ms = -335.2 ms (-52.9 %)**, interleaved A/B **+324.1 ms**.
Unlike WP-A16 §2.6's pass 1, the *bare* median moved in the right direction,
so no load-correction argument is needed.

Named cumulative rows (`python -X importtime`, medians of 5):

| module | BEFORE | AFTER |
|---|---|---|
| `lumenairy` | 622.7 ms | **280.3 ms** |
| `lumenairy.propagators.fft_infra` | 415.5 ms | **50.2 ms** |
| `scipy.fft` | 396.0 ms | *(not imported)* |
| `scipy` | 27.6 ms | 27.6 ms |
| `scipy.special` | 49.0 ms | *(not imported)* |
| `numpy.f2py` | 108.3 ms | *(not imported)* |
| `numpy.testing` | 96.6 ms | *(not imported)* |
| `charset_normalizer` | 77.2 ms | *(not imported)* |

Against WP-A16's expectation ("the bulk of the ~540 ms A15b's estimate
attributed to this pair"): the realised figure is 324–335 ms, and the rows
above say where the rest went — `charset_normalizer` and part of
`numpy.testing` are reached by other importers too, so they were never
`scipy.fft`'s to give back.

**Gate — bit-identity.**  All seven FFT/ASM/propagation files pass:

```
tests/unit/test_perf_v4_12_0_fft_infra.py                11
tests/unit/test_audit_w2_fft_state.py                     5
tests/unit/test_audit2609_a5_verify_fft_buffer_threads.py 2
tests/unit/test_audit2609_a15b_optional_and_knobs.py     33
tests/unit/test_v5_2_walker_pep562_forwarding.py          4
tests/unit/test_propagation.py                           10
tests/unit/test_audit2609_a17_history_relocation.py     229
-> 248 passed  (46 failures, ALL in the A17 file, all on documents a
   concurrent sweep is mid-writing — see §4)
```

Zero non-A17 failures.  `test_v5_2_walker_pep562_forwarding.py` in particular
is green, which is the check that design point 1 above actually worked.

### 2.9 (item 10) `scripts/record_history_fingerprints.py`

**The gap.**  WP-A17 §2.7 named it as a residual risk: the recorded
fingerprints pin a module "forever, which is the point but also means a
deliberate code change must re-record the two hashes in the same commit".
There was no way to do that except by hand-editing a hash — and a hash that
gets hand-edited records nothing.

**The tool.**  `record_history_fingerprints.py <module-or-doc> ...` accepts any
of the four spellings a caller has to hand (module path, dotted module,
document stem, document path; no argument = every document), and:

* computes both fingerprints by **importing the checker's own**
  `ast_fingerprint` / `token_fingerprint` by path.  A second implementation
  would be two definitions of "unchanged" that agree until the day they do
  not, at which point the recorder writes a hash the gate rejects;
* `--check` reports drift per document with both digests and exits 1, writing
  nothing;
* a write requires `--reason`, which is appended to the header as
  `re_recorded: <date> -- <reason>`.  The rewrite is **in place, line by
  line**, not a regenerated header: the header also carries
  `pre_relocation_lines` (load-bearing for the checker's table-of-contents
  test), `recorded_by` and `checker`, and a regenerating writer would have to
  know about every field a later work package adds;
* derives a document's repository root from the document's own path
  (`<root>/docs/history/<doc>.md`), so it can be pointed at a copy of the tree
  — which is how its own test exercises it without touching real documents;
* a malformed document is reported and the sweep CONTINUES.  Over a directory
  the useful output is every problem, not the first one.

**Used for item 9.**  After the `fft_infra.py` change:

```
DRIFT fft_infra.md   lumenairy/propagators/fft_infra.py  (ast, token)
        recorded ast   1a11380b...ec33      live ast   bd5002a4...6eab
        recorded token c4f06ed7...9299      live token 5a24405d...1c40
        re-recorded, reason: scipy.fft deferred behind find_spec + a first-use
                             accessor; no behaviour change (WP-A22 item 9)
```

and the header now carries the new digests plus that `re_recorded:` line.  The
A17 gate's 18 ids for `carrier` / `carrier_field` / `fft_infra` are green.

**The rule is written down in three places** so it is not folklore:
`CONTRIBUTING.md` gained a "Modules with a history document" section with the
commands (and a bullet against version-history narrative in the source); the
checker's module docstring gained a "THE MAINTENANCE RULE" paragraph; and both
of the checker's fingerprint assertions now print the exact recorder command
with the module already substituted in, plus "do not hand-edit the hash".

**Gate: `tests/unit/test_audit2609_a22_history_fingerprint_tool.py`** (11 ids,
2.5 s).  A recorder is a tool that can silently weaken another gate, which is
the one kind of tool that must not be trusted on inspection, so:

| what | why it is not redundant |
|---|---|
| the recorder's **AST** defines neither fingerprint function and does load the checker | cannot be caught by running the tool: two implementations agree until they do not |
| `--check` is byte-compared before/after | it is specified read-only so CI can run it on a dirty tree |
| a value-preserving re-spelling (`5` → `0x5`) is still drift | the AST folds both spellings; this is the whole reason two fingerprints exist.  The test first ASSERTS the AST fingerprint did *not* move, so the arm cannot pass vacuously |
| re-recorded hashes equal the **checker's** reading of the drifted source | "the tool now says OK" would also be said by a tool that wrote one wrong value into both places |
| every other header field survives, and the prose below it | the in-place rewrite's actual risk |
| two re-records leave **two** trail lines | the trail is the only thing distinguishing a maintained baseline from a silenced gate |
| a write with no `--reason` exits 2 | a re-record with no reason is indistinguishable from silencing |
| one malformed document does not abort the sweep | else the state of every other document goes unreported |
| the recorder's reading of the REAL tree agrees with the gate's | if one passes and the other fails, every re-record since is suspect |

---

## 3. Files touched

**New**

* `scripts/record_history_fingerprints.py`
* `tests/unit/test_audit2609_a22_history_fingerprint_tool.py`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A22_REPORT.md` (this file)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A22_CHANGELOG.md`

**Modified** (all within the ownership list)

* `pyproject.toml` — the F541 per-file ignore deleted; `[tool.mypy] files`
  **unchanged** (item 7 did not qualify).
* `lumenairy/propagators/fft_infra.py` — `scipy.fft` probe + accessor + module
  `__getattr__`; four readers; one `__all__` entry.  **One change, item 9.**
* `docs/history/fft_infra.md` — header only (two digests + one `re_recorded:`
  line), written by the new recorder.
* `tests/unit/test_audit_except_budget.py` — census entry deleted, four stated
  numbers re-measured.
* `tests/unit/test_ci_kernel_consistency.py` — the wall-clock assertion
  replaced; two new module constants; the BUDGET docstring paragraph.
* `tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py` — x64 at import + a
  module-scoped autouse fixture.
* `tests/unit/test_audit2609_a17_history_relocation.py` — the maintenance rule
  in the docstring; the recorder command in both fingerprint failure messages.
* `scripts/check_doc_identifiers.py` — four triaged exclusions deleted.
* `Migration-Guide.md` — the dead class spelling; three late-package notes.
* `README.md` — two "Start here" rows; one real-lens section pointer.
* `CONTRIBUTING.md` — "Modules with a history document"; one Avoid bullet.

**NOT touched, deliberately**: `lumenairy/elements/_lens_thin.py` and
`lumenairy/elements/pmm/stack2d_pure.py` (§5.1, §5.2 — both were clean when
this work package started and are `M` now);
`tests/unit/test_audit2609_a15a_packaging.py` (item 7 did not qualify, so the
ratchet was not raised); `tests/unit/test_audit2609_a21_doc_identifiers.py`
(its ceiling is a `<=` bar and only shrank).

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`,
`-q --no-header -p no:cacheprovider`.

| command | result |
|---|---|
| `test_audit_except_budget.py` | **4 passed**, 0.37 s |
| `test_audit2609_a22_history_fingerprint_tool.py` | **11 passed**, 2.50 s |
| `test_audit2609_a21_doc_identifiers.py` | **5 passed**, 7.72 s |
| `test_v4_14_0_dispatcher_pin_apply_lens.py` (ALONE) | **35 passed**, 6.55 s (was 12 failed / 23 passed) |
| `test_v4_14_0_... test_v5_12_0_audit_fixes.py` (both orders) | **40 passed**, 14.2 / 14.5 s |
| 7 FFT/ASM/propagation files + A17 | **248 passed, 46 failed**, 15.5 s — every failure in the A17 file, see below |
| `test_audit2609_a17_history_relocation.py -k "fft_infra or carrier"` | **18 passed**, 4.4 s |
| `test_audit2609_a21_pmm_warning_filter.py` + `test_ci_kernel_consistency.py` | **11 passed, 1 failed** — the failure is pre-existing, see below |
| `test_audit2609_a8_thin_elements.py` + `test_audit2609_a8_verify.py` | **97 passed**, 15.5 s (item 8 baseline) |
| `test_audit2609_a15a_packaging.py`, `test_v5_1_0_agent_c_split.py`, `test_v4_16_2_dispatcher_pin_doc_consistency.py` | **141 passed**, 1.89 s |
| `test_niche_p8_capstone.py`, `test_v4_15_agent_f.py`, `test_v4_16_2_agent_d.py`, `test_v5_2_walker_changelog_changeset.py` | **35 passed, 2 skipped**, 131.7 s |
| `test_niche_p8_capstone.py` … + packaging batch (first run) | **176 passed, 2 skipped**, 128.2 s |
| `ruff check lumenairy tests scripts` | **All checks passed** |
| `mypy` (the whitelist) | **2 errors**, both pre-existing, see below |
| `python -c "import lumenairy"` | **OK**, 5.45.1 |
| `python scripts/check_doc_identifiers.py` | **OK**, denominator 597, unresolved 0 |
| `python scripts/record_history_fingerprints.py --check` | **OK** on all 81 documents present at close |

**Final consolidated run** of the touched files, after every edit was in place
(the A17 file excluded — see §4(b) — and `test_ci_kernel_consistency.py`
excluded, §4(a)):

```
tests/unit/test_audit2609_a22_history_fingerprint_tool.py
tests/unit/test_audit_except_budget.py
tests/unit/test_audit2609_a21_doc_identifiers.py
tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py
tests/unit/test_v5_2_walker_pep562_forwarding.py
tests/unit/test_perf_v4_12_0_fft_infra.py
tests/unit/test_audit_w2_fft_state.py
tests/unit/test_audit2609_a5_verify_fft_buffer_threads.py
tests/unit/test_audit2609_a15b_optional_and_knobs.py
tests/unit/test_propagation.py
-> 120 passed, 5 warnings in 20.63s
```

and the structural claim item 9 exists for, re-checked at close:

```
import OK 5.45.1 | scipy.special loaded: False | scipy.fft loaded: False
```

### Pre-existing failures found, NOT caused by this work package

**(a) `test_ci_kernel_consistency.py::test_this_arm_agrees_with_the_committed_census`
— RED, and it is a real finding for the census's owner.**

It fails at step **1**, the RULE check, which this work package did not touch;
my new block is step 4.  Attribution, measured rather than argued: the **HEAD
blob** of that same file, executed against this tree, fails identically —

```
HEAD VERSION RESULT: FAILED
first line: arm 'WIN-unknown-t1' takes 2 guard decision(s) the RULE does not
            permit for the answer class measured HERE ...
```

The violations:

```
pmm1d_interface/warned@1e-04  class=correct  decision=warn  permitted=['silent']
pmm1d_interface/warned@1e-05  class=correct  decision=warn  permitted=['silent']
```

and the classes this arm measures are `correct` on **all eight** rows,
including `sliver/pmm1d@1e-05` — where the census records every measured arm
as `wrong` and only the transcribed CI arm as `correct`.  Identical with
`OPENBLAS_CORETYPE=HASWELL` pinned, so it is not a kernel effect.

**Reading:** a PMM 1-D change landed during this campaign (commit `56a76f22`,
WP-A12, touches `pmm/oned.py` / `pmm/_core.py` — "lazy sliver arbiter", among
others) has made this box answer the ill-conditioned interface fixtures
CORRECTLY.  That is very likely good news, but it invalidates the committed
census and it is exactly the divergence the premise gates in six test families
rest on.  The census needs re-measuring by its owner —
`probe_decisions.py` on each arm then `merge_arms.py`, per the module
docstring — and the `warned@*` rows re-argued (a correct answer that warns is
a rule violation as the table stands).  Note that
`test_the_census_carries_the_ci_arm_that_answers_these_fixtures_correctly`
still PASSES, because it reads the committed table rather than the live arm;
the divergence it protects is now stale in fact but not in the file.

My step-4 block was verified separately, with the pre-existing step-1/step-2
assertions neutralised in process:

```
WP-A22 BUDGET BLOCK: PASSED -- probed 24 decisions, 4 hypotheticals,
                              0 expensive, wall 1.56 s
```

**(b) `test_audit2609_a17_history_relocation.py` — 46 ids RED, all transient.**

Every failure is a parametrisation over a document that is **untracked**
(`git status` → `??`) and being written by a concurrent history-relocation
sweep right now — `lumenairy.propagators.asm`, `.mhs`, `.system`,
`lumenairy.analysis.ao`, … — failing on
`test_the_header_names_a_real_module_and_two_fingerprints` (header still
incomplete) and `test_the_source_still_points_at_the_history_document` (the
pointer comment not yet added).  The three **tracked** documents
(`carrier.md`, `carrier_field.md`, `fft_infra.md`) pass all 18 of their ids.
Not mine, and it will clear as those sweeps finish.

**(c) `mypy` on the whitelist — RED, 2 errors, unchanged since WP-A21 §5.1.**

```
lumenairy\backend\__init__.py:56: error: Function is missing a type annotation  [no-untyped-def]
lumenairy\backend\__init__.py:79: error: Function is missing a return type annotation  [no-untyped-def]
Found 2 errors in 1 file (checked 30 source files)
```

`lumenairy/backend/` has been on the whitelist since v5.0; the lazy `scipy`
forward that landed there is unannotated.  The file is clean in `git status`
and is not mine.  This is a **merge-blocking** gate
(`unit-tests.yml`'s mypy job runs `continue-on-error: false`), so it needs an
owner before the campaign closes — exact patch in §5.3.

---

## 5. Requested changes outside my ownership

### 5.1 BLOCKED — `lumenairy/elements/_lens_thin.py`: break the fourth lens 2-cycle

`git status --short` showed this file **clean** at the start of this work
package, **`M`** when I reached the item, and **`M`** again at close, so per my
brief it was not touched.  The patch below is verified; it needs applying by
whoever holds the file, or by me once it is committed.

**Verification, done on an overlay copy so the repo file was never written.**

*Cycle count*, module-level-only AST walk over the lens family (imports inside
a `def` are deferrals, not cycles; walking into function bodies is what makes
a naive walker report 7).  The unpatched tree reproduces WP-A16 §2.4 exactly:

```
BEFORE   module-level 2-cycles: 4        AFTER (overlay)  module-level 2-cycles: 3
  _lens_real  <-> lenses                   _lens_real  <-> lenses
  _lens_thin  <-> lenses     <-- gone      _lens_traced <-> lenses
  _lens_traced <-> lenses                  lenses <-> lenses_maslov
  lenses <-> lenses_maslov
```

*Bit-identity*, patched module loaded under the same dotted name in a live
interpreter and compared against the shipped one on a fixed complex field
(`N=96`, seeded RNG):

```
apply_thin_lens          bit-identical: True   complex128
apply_spherical_lens     bit-identical: True   complex128
apply_aspheric_lens      bit-identical: True   complex128
apply_cylindrical_lens   bit-identical: True   complex128
apply_grin_lens          bit-identical: True   complex128
apply_axicon             bit-identical: True   complex128
apply_thin_lens (c64)    bit-identical: True   complex64
CUPY_AVAILABLE parity: True | cp forward -> None (both) | AttributeError on unknown name: OK
```

Baseline for the named gates on the unpatched tree:
`test_audit2609_a8_thin_elements.py` + `test_audit2609_a8_verify.py` = **97
passed**, 15.5 s.

**THE PATCH.**  Anchored on content, not line numbers — the sweep is moving
lines under it (the head block was at 34-62 when this work package started and
is at 36-63 now).  Replace the head block:

```python
# CuPy is lazy-loaded; this module accesses it via the lenses module's
# lazy slot so a single load is shared across the package.
from . import lenses as _lenses_module
from .lenses import (
    CUPY_AVAILABLE,
)


def _is_cupy_array(x):
    return _lenses_module._is_cupy_array(x)


# Module-level cp alias.  Updated whenever _lenses_module's cp is loaded
# (it points at None until first GPU call, then the actual cupy
# module).  We sync via a property-style accessor below.

def __getattr__(name):
    """PEP 562 module-level __getattr__: route ``cp`` to the lenses
    module's lazy slot.  Triggers when callers do
    ``from ._lens_thin import cp`` -- the in-function references inside
    each apply_* below resolve via this fallback if `cp` isn't yet a
    module global.
    """
    if name == 'cp':
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        return _lenses_module.cp
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
```

with:

```python
# CuPy is lazy-loaded through the shared optional-dependency helper.  That
# helper is a LEAF -- nothing in ``lumenairy.backend`` imports back into
# ``elements`` -- so reaching it directly rather than through ``.lenses``
# keeps this module out of a module-level import cycle with ``lenses``,
# which still re-exports every name below.  The lazy slot is the SAME one
# ``lenses.cp`` reads, so a single CuPy import is shared across the package.
from ..backend._optional import (
    CUPY_AVAILABLE,
    ensure_cupy as _ensure_cupy,
    is_cupy_array as _is_cupy_array,
)


def __getattr__(name):
    """PEP 562 module-level __getattr__: route ``cp`` to the shared lazy
    slot.  Triggers when callers do ``from ._lens_thin import cp`` -- the
    in-function references inside each apply_* below resolve via this
    fallback if `cp` isn't yet a module global.
    """
    if name == 'cp':
        return _ensure_cupy()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
```

and replace **all six** occurrences of

```python
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
```

with

```python
        _cp = _ensure_cupy()
```

(six was the count when this was verified; `grep -c '_lenses_module.cp$'` reads
8 now because two of the hits are the comment lines named below.  Replace every
occurrence of the three-line block, whatever the count.)

Then the four comments that name the old mechanism (`the ``_lenses_module.cp``
rationale` ×3, and the `_lenses_module lazy slot rather than a bare global`
block above `apply_grin_lens`) should say "the shared lazy slot" instead.
After that, `grep -c _lenses_module lumenairy/elements/_lens_thin.py` must
return **0**.

**Two knock-on notes for the same commit.**

* `lumenairy/elements/lenses.py:146` currently says
  "``_lens_thin._is_cupy_array`` delegates here, so this is also the answer
  …".  After the patch it delegates to `backend/_optional.is_cupy_array`
  (the same function `lenses` itself now uses), so that sentence needs one
  word changed.  `lenses.py` is not mine.
* `lenses.py:1076`'s `from ._lens_thin import (...)` re-export is unaffected —
  the cycle broken is the back-edge, not the forward one.

### 5.2 BLOCKED — `lumenairy/elements/pmm/stack2d_pure.py:1418`: one keyword

Same situation: clean at the start of this work package, `M` when I reached
the item and `M` at close.  WP-A21 §5.3's request, verified on this build.

**THE PATCH**, at `stack2d_pure.py:1418-1419` (the only
`check=("warn",))` in the file, inside `_warn_stag_shared_redundancy`):

```python
-        _validate_stag_cost("PMM2DStackPure.solve", int(self.M), *cells,
-                            check=("warn",))
+        _validate_stag_cost("PMM2DStackPure.solve", int(self.M), *cells,
+                            check=("warn",), stacklevel=4)
```

**Verified** by probing every frame at the raising site during a real deferred
warn, with `warnings.warn` instrumented so no frame was added or removed:

```
shipped call site passes stacklevel = 3
  stacklevel=1: twod_staggered.py  _validate_stag_cost            :628
  stacklevel=2: stack2d_pure.py    _warn_stag_shared_redundancy   :1418
  stacklevel=3: stack2d_pure.py    solve                          :1441  <-- TODAY
  stacklevel=4: <user's file>      user_call                      :28    <-- the patch
  stacklevel=5: <user's module>
```

So 4 is the value, and it is the caller's own `st.solve()` line.  The default
is unchanged, so nothing breaks until it lands.

**The test extension that must land in the SAME commit** (it fails before the
patch and passes after), appended to
`tests/unit/test_audit2609_a21_pmm_warning_filter.py`:

```python
def test_the_deferred_path_reports_at_the_callers_line():
    """The shared-stack entry must point at the user's ``solve()``.

    The direct path already does (``test_the_warning_is_attributed_to_the_
    caller_on_the_direct_path``); the deferred one reaches the warning through
    ``PMM2DStackPure._warn_stag_shared_redundancy``, one frame further down,
    so it needs ``stacklevel=4`` at ``stack2d_pure.py:1418``.  Bar: an exact
    filename.  A user who cannot see WHICH of their calls is expensive cannot
    act on advice about a ~1000x cost cliff.
    """
    import lumenairy.elements.pmm.stack2d_pure as _sp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _deferred()
    hits = [w for w in caught if 'SEGMENT grid' in str(w.message)]
    assert len(hits) == 1, [str(w.message)[:60] for w in caught]
    assert hits[0].filename == __file__, (
        f'the deferred advice is reported at {hits[0].filename} '
        f'(the library\'s own line) rather than at the calling line in '
        f'{__file__}.  Pass stacklevel=4 at {_sp.__file__}:1418.')
    assert hits[0].message.args[0].startswith('PMM2DStackPure.solve:')
```

(`_deferred` and the imports it needs are already in that file.)  MEASURED
against today's tree: `hits[0].filename` is `stack2d_pure.py`, so the
assertion fails before the patch, as a FAIL-BEFORE test must.

Note that `test_a_module_scoped_filter_on_this_module_does_not_catch_it`
stays green either way — the module filter is on `twod_staggered`, and moving
the attribution from `stack2d_pure` to the caller does not touch it.

### 5.3 `lumenairy/backend/__init__.py` (WP-A16 / the backend sweep) — two annotations, MERGE-BLOCKING

Still open from WP-A21 §5.1, and it is the whole reason `mypy` is red:

```python
-def __getattr__(name):
+def __getattr__(name: str) -> Any:
...
-def __dir__():
+def __dir__() -> list[str]:
```

with `from typing import Any` added.  The mypy job is
`continue-on-error: false`, so the campaign cannot close with this open.

### 5.4 Root `lumenairy/__init__.py` / `lumenairy/elements/lenses.py` — the 31 re-exports

To make the root whitelist-able (item 7), the 30 names at `__init__.py:149`
need to be in `elements/lenses.py`'s `__all__` (or spelled `from X import Y as
Y` there), `trace_world` likewise in `raytrace`'s, and the root's own PEP 562
pair at `:2192` / `:2220` annotated as in §5.3.  Full list in §2.7.  When that
lands, `"lumenairy/__init__.py"` joins `[tool.mypy] files` and the ratchet in
`test_audit2609_a15a_packaging.py::test_the_mypy_whitelist_only_grows` goes
**25 → 26** — one change, both halves, or the ratchet pins a red module.

### 5.5 `validation/probe_ci_kernel_sweep/` owner — the census is stale in fact

See §4(a).  This box now answers the ill-conditioned 1-D interface fixtures
CORRECTLY where the committed census records every measured arm as `wrong`,
and two `warned@*` rows violate the rule table as a result.  Re-measure the
arms and re-argue those rows; the premise gates in six test families cite the
divergence this invalidates.

### 5.6 `lumenairy/elements/lenses.py` — one stale sentence

`lenses.py:146`, after §5.1 lands.  Detail in that section.

### 5.7 Orchestrator — the A17 gate will stay noisy until the sweeps land

46 ids are red on untracked, mid-write history documents (§4(b)).  Nothing to
do beyond sequencing: the sweeps' own commits will clear them.  Worth running
`python scripts/record_history_fingerprints.py --check` at campaign exit —
it is the fastest way to see whether any relocated module drifted without a
re-record, and it now reports every document in one pass.

---

## 6. Deferred, with designs

1. **The remaining three lens 2-cycles** (`_lens_real <-> lenses`,
   `_lens_traced <-> lenses`, `lenses <-> lenses_maslov`).  Unchanged from
   WP-A16 §6 item 1: they need the `elements/_lens_kernels.py` extraction
   (`surface_sag_general`, `surface_sag_biconic`,
   `_warn_if_aperture_exceeds_grid`, `_fit_normaliser`,
   `_multi_indices_total_degree`, `NUMEXPR_AVAILABLE` +
   `_ensure_numexpr_loaded`), which the audit budgets at ~3 d and which should
   follow WP-A17 part 2 rather than collide with it.  The module-level-only
   cycle walk used here is ~60 lines and would make a good permanent gate once
   the count is where it should be — assert the count, listing the survivors,
   so a NEW cycle fails even while the old ones stand.  **Effort ~1 h** for the
   gate, ~3 d for the extraction.
2. **`test_ci_kernel_consistency.py`'s `_RETAKE_DECISIONS = 24`** is derived
   from the census at run time and cross-checked against the constant, but the
   constant still has to be edited when the table is regenerated.  A cleaner
   form records the per-section counts in `decisions.json` itself at merge
   time, so the test reads them.  That is a `merge_arms.py` change, which is
   the census owner's file.  **Effort ~1 h.**
3. **`--check` in CI.**  `record_history_fingerprints.py --check` and
   `check_doc_identifiers.py` both exit 0/1, need no network, and take ~1 s and
   ~4.5 s.  Both are already covered by unit tests, so a separate `docs` job is
   optional — but it would give a clearer signal than a unit-test failure, and
   the same shape as the existing `check_source_line_citations.py` wiring.
   I own no workflow file.  **Effort ~30 min.**
4. **`lumenairy/backend/scipy` and the remaining import cost.**  After item 9,
   `import lumenairy` is 280 ms cumulative, of which `lumenairy.elements` is
   still the largest block.  The next measurement worth taking is an
   `-X importtime` tree of the AFTER state rather than a guess; the two obvious
   remaining candidates (`scipy` itself at 27.6 ms via the `find_spec` probe,
   and whatever else pulls `numpy` submodules) are both much smaller than what
   was just removed, so this is no longer a headline item.  **Effort ~2 h to
   measure.**

---

## 7. Path to the changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A22_CHANGELOG.md`

---

# Follow-up (after the elements sweep landed)

The first pass shipped as `7d03d799`.  The coordinator then confirmed that
`lumenairy/elements/_lens_thin.py` and `lumenairy/elements/pmm/stack2d_pure.py`
were clean, added the three items WP-A17 sweep 2 could not touch because they
are CODE rather than comments (`fixes/WP-A17_SWEEP2_REPORT.md` sec. 7 items
2-3), and later added sweep 1's sec. 9 item 4 (the JAX-twin dtype fill).  Every
module's fingerprints were re-recorded in the same change as its code edit.

## F1. Summary

| item | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| 8a. `_lens_thin` 2-cycle (was BLOCKED) | **fixed** | `lumenairy/elements/_lens_thin.py:36-57` + six call sites + five comments; `lumenairy/elements/lenses.py:146-150` | `test_audit2609_a8_thin_elements.py`, `test_audit2609_a8_verify.py` (97 passed) | module-level-only AST cycle walk | **4 -> 3** module-level 2-cycles |
| 8b. `stack2d_pure` stacklevel (was BLOCKED) | **fixed** | `lumenairy/elements/pmm/stack2d_pure.py:1418-1424` | `test_audit2609_a21_pmm_warning_filter.py` **6 -> 7 ids** | frame probe at the raising site | advice reported at `stack2d_pure.py:1446` -> **at the caller's `solve()`** |
| F-a. `eme_diffraction` refusal message | **fixed** | `lumenairy/elements/eme/eme_diffraction.py:166-170` | 271 passed across the EME/thin batch | the message text itself | narrative parenthetical -> present tense; retired wording in the history doc at L167-169 |
| F-b. `pmm/stack.py` refusal message | **fixed** | `lumenairy/elements/pmm/stack.py:3106-3116` | `test_audit2609_a12_*` / `a13_*` (105 passed) | the message text itself | self-retraction removed; retired wording at L3209-3215 |
| F-c. `_ARCHIVE_SLANT_FOLD` moved out | **fixed** | `lumenairy/elements/pmm/_core.py:7466-7472` (was `:7466-7523`), `:6649`, `:6681` | same PMM files | AST diff of module-level names | **one** module-level name removed, `__all__` byte-identical |
| F-d. `asymptotic_jax_twin` dtype fills | **fixed** | `lumenairy/propagators/asymptotic_jax_twin.py:524-532` | `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` **239 -> 247 ids**; the asymptotic files | jnp dtype promotion under `jit` | pin **1 failed / 238 passed -> 0 failed / 247 passed**; `safe_phi` float32 -> **complex64** becomes float32 -> **float32**; float64 output bit-identical |
| F-e. the pin could not see continuation lines | **fixed** | `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py:80, 126-143, 147-196, 269-282, 497-568` | the file itself | the regex vs the AST walk on eight synthetic sources | regex sees **0** of the wrapped form, the walk sees **1** |
| F-f. fingerprints re-recorded | **done** | six `docs/history/*.md` headers | `test_audit2609_a17_history_relocation.py` (708 passed) | the recorder + the checker | 6 drifted -> **0**; each carries a `re_recorded:` line with its reason |
| F-g. the A17 falsifiability arm was RED | **fixed** | `tests/unit/test_audit2609_a17_history_relocation.py:335-352, 419-460` | that file | mutation 3 on `lumenairy/_context.py` | `no small integer literal found to re-spell` -> **116 passed** |

## F2. Per item

### F2.1 (8a) The `_lens_thin <-> lenses` cycle

Applied as sec. 5.1 specified, with one deviation the linter forced and one
knock-on the report predicted.

* **Deviation.**  The parenthesised `from ..backend._optional import (...)`
  form trips `I001`: this repo's isort profile wants one name per line here,
  and `lenses.py:47-49` already imports the same three names that way.  It is
  spelled to match, and the block moved ABOVE `..glass` so the run is sorted:

  ```python
  from ..backend._optional import CUPY_AVAILABLE
  from ..backend._optional import ensure_cupy as _ensure_cupy
  from ..backend._optional import is_cupy_array as _is_cupy_array
  from ..glass import get_glass_index  # 4.10: was missing, broke apply_axicon
  ```

  Adding `_lens_thin.py` to the `I001` per-file ignore would also have worked
  and was not done: that list is the one WP-A15a wrote "do not grow it" on,
  and matching the existing spelling costs nothing.
* **Six** three-line `_lenses_module.cp` resolution blocks became
  `_cp = _ensure_cupy()`, and **five** comments naming the old mechanism were
  reworded.  `grep -c _lenses_module lumenairy/elements/_lens_thin.py` now
  returns **0**.
* **Knock-on, as predicted:** `lenses.py:146`'s "``_lens_thin._is_cupy_array``
  delegates here, so this is also the answer the thin-lens family gets" was
  true and is not any more.  Reworded (the one comment the coordinator
  authorised) to say that `_lens_thin` asks `backend._optional.is_cupy_array`
  directly, takes the same answer from the same helper, and that the extra
  short-circuit here is about call cost rather than about the answer.  That is
  a DOCSTRING edit, so it moved neither of `lenses.py`'s fingerprints -- the
  recorder listed `lenses.md` as clean, which is the checker doing its job.

MEASURED, module-level-only AST cycle walk over the lens family:

```
BEFORE  module-level 2-cycles: 4      AFTER  module-level 2-cycles: 3
  _lens_real   <-> lenses               _lens_real   <-> lenses
  _lens_thin   <-> lenses   <-- gone    _lens_traced <-> lenses
  _lens_traced <-> lenses               lenses <-> lenses_maslov
  lenses <-> lenses_maslov
```

Behaviour: `CUPY_AVAILABLE` unchanged, `_lens_thin.cp` still forwards through
PEP 562 to the shared lazy slot, an unknown attribute still raises
`AttributeError`, and `lenses.apply_thin_lens is _lens_thin.apply_thin_lens`.
Bit-identity of all six entry points plus a complex64 case was measured before
the patch was proposed (sec. 5.1); the applied text is that same patch.

### F2.2 (8b) The deferred PMM advisory's `stacklevel`

`stacklevel=4` applied, with a why-comment naming the extra frame.

**FAIL-BEFORE, by reverting the change in process** (rebuilding
`_warn_stag_shared_redundancy` with the pre-fix call and re-running the same
fixture, so no frame is added or removed):

```
AFTER        reported at <caller>:19
BEFORE       reported at stack2d_pure.py:1446
BEFORE is the library file : True
AFTER  is the caller file  : True
```

The new pin, `test_the_deferred_path_reports_at_the_callers_line`, asserts the
exact filename and that the message still opens `PMMStack.solve:`.  That file
goes **6 -> 7 ids, all passing**.  The two-edged counter-pin beside it
(`test_a_module_scoped_filter_on_this_module_does_not_catch_it`) stays green,
as sec. 5.2 predicted: the module filter is on `twod_staggered`, and moving the
attribution from `stack2d_pure` to the caller does not touch it.

### F2.3 (F-a, F-b) Two narratives inside executed strings

Both were history *inside a string literal the interpreter runs*, which is why
a documentation-only sweep could not take them: changing one moves both
fingerprints.

**`eme_diffraction.py`** -- the zero-norm refusal said the failure "used to
surface as an opaque 'SVD did not converge' LinAlgError".  Now: "Unguarded,
such a column reaches the least-squares solve and fails there as an opaque 'SVD
did not converge' LinAlgError that names neither Psi nor the column."  The
connection is kept, in the present tense, because a reader with an old
traceback still needs it; what went is the claim about when the library
changed.  `zero-norm or non-finite` is untouched -- `test_niche_audit_w6_eme.py:783`
matches on that phrase (checked before editing).

**`pmm/stack.py`** -- the per-layer `stabilize='slices'` refusal quoted and
retracted its own earlier wording ("this message used to say per-layer grids
have 'no cross-layer walls to perturb', which is WRONG").  Now it states the
corrected fact positively: "min_feature IS live on this path even so: a window
is itself a union and contains the adjacent-slice collisions."  Every operative
instruction survives.  Grepped first: no test matches the retracted sentence.

Both retired texts are reproduced verbatim in their history documents under
their PRE-RELOCATION line numbers (L167-169 and L3209-3215, read off
`2622449f~1` rather than guessed), each with a note saying why it was moved
separately from the sweep.

### F2.4 (F-c) `_ARCHIVE_SLANT_FOLD`

**Grepped first, as instructed.**  Across `*.py`, `*.md`, `*.toml`, `*.cfg`,
`*.txt` and `*.yml`, the only hits outside `docs/audits/` were the assignment
itself and **two comments that merely name it** (`_core.py:6648`, `:6679`).  No
code, no test, no `__all__` entry: it was parsed and bound on every
`import lumenairy` and read by nothing.

**Correction to the sweep report's figure.**  Sweep 2 sec. 7 item 3 gives
"lines 7472-7648, ~180 lines".  MEASURED from the AST, the assignment spans
**7516-7565 pre-relocation / 7474-7523 today = 50 lines** (58 including the
ARCHIVE section header above it).  The ~180 appears to have been measured to
the end of file rather than to the end of the statement.  It is still the
largest single history block in the partition; it is not 180 lines.

The string body moved verbatim into
`docs/history/lumenairy.elements.pmm._core.md` under a new `### L7516-7565`
section, the two pointer comments now name that document instead of the
constant, and the ARCHIVE section header stays but says the record lives in the
document -- with the reason the document is the better home: it is pinned to
the module by the A17 checker, which a module-level string nobody reads is not.

Verified inert by diffing module-level names against `HEAD`:

```
module-level names removed: ['_ARCHIVE_SLANT_FOLD']
module-level names added  : []
counts: 183 -> 182
__all__ identical: True | entries: 107
```

### F2.5 (F-d) `asymptotic_jax_twin.py` -- the dtype-matched fills

**FAIL-BEFORE:** `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` reported
**1 failed / 238 passed**, naming `propagators/asymptotic_jax_twin.py`.

**What is actually wrong, measured rather than assumed -- and it is not quite
what sweep 1 sec. 9 item 4 says.**  That report states the literal "forces a
complex128 promotion regardless of the iterate's dtype".  On this JAX build it
does not: a Python `complex` is WEAKLY typed, so a complex64 operand stays
complex64 with x64 on or off.  What the literal does force is a **real ->
complex** promotion.  Measured with `jax.jit`, x64 enabled:

| operand | old `0.0 + 0.0j` | new `zeros((), dtype)` |
|---|---|---|
| `b_quad` complex64 | complex64 | complex64 |
| `b_quad` complex128 | complex128 | complex128 |
| `phi_star` float32 | **complex64** | **float32** |
| `phi_star` float64 | **complex128** | **float64** |

and `phi_star` at the site IS real -- measured `float64` on the x64 path,
straight out of `_compute_M_b_xp`.  So:

* `safe_bquad` (the site the pin caught) was **inert** on this build: `b_quad`
  is complex already.  It is still the forbidden spelling, and it depends for
  its harmlessness on weak-typing rules JAX has changed before.
* `safe_phi` (the site the pin MISSED) was the live one: it silently returned a
  complex phase whose imaginary part was always zero, and at float32 it
  produced a complex64 array where a float32 one was asked for.

The irony is worth recording: the sibling that escaped the pin on a line-break
technicality was the one that was actually wrong.

**Values are preserved** (`jnp.all(old == new)` True on both branches at both
precisions), and the end-to-end float64 output is **bit-identical**: the BEFORE
arm was produced by loading the HEAD blob of the module under its own dotted
name and running `propagate_modal_asymptotic_lg00_jax` on the Y3 9x9 fixture --

```
BEFORE dtype complex128 | AFTER dtype complex128
BIT-IDENTICAL (np.array_equal): True
max|diff| = 0.0
```

**Not changed, and the distinction is the point.**  `safe_det = jnp.where(
ok_det, det_M, 1.0 + 0.0j)` two lines above is the same SHAPE but a different
defect class: the fill is a NON-ZERO sentinel whose value is load-bearing (it
keeps `1.0 / safe_det` finite), so `zeros((), dtype)` is not its migration --
substituting one would change the answer.  The same holds for
`propagators/asymptotic.py:658` and five sites in `elements/_lens_traced.py`.
A first draft of the walk below flagged all six; the rule was made
value-correct instead.

### F2.6 (F-e) The pin could not see a continuation line

The scanner was per LINE and the pattern required the call and its fill on the
same line, so a wrapped call was invisible:

```python
safe_phi = jnp.where(jnp.isfinite(jnp.abs(phi_star)), phi_star,
                     0.0 + 0.0j)
```

That is not hypothetical -- it is the `safe_phi` above, two lines below a site
the pin DID catch, hidden from v4.14 to v5.45 by nothing but where the line
broke.

A **structural pass** now runs beside the regex and the two are UNIONED, so the
regex's exact behaviour is preserved and the walk only ever adds sites it was
blind to.  It finds every `*.where(cond, value, <literal complex zero>)` call
regardless of line breaks, using `ast.literal_eval` on the fill so `0j`,
`0.0j` and `0.0 + 0.0j` are all one rule, and it reports the FILL's own line so
the failure points at the literal.  Being structural, it cannot match inside a
comment or a string at all.

Two exemptions, both stated rather than heuristic:

* a NON-ZERO sentinel is not a hit (`literal_eval` gives `(1+0j) != 0`);
* a call whose VALUE branch is itself a literal complex constant is not a hit:
  the array is complex by construction and there is no operand dtype to
  preserve.  `doe.py:534`'s `np.where(is_even & inside, 1.0 + 0j, 0.0 + 0j)`
  binary zone mask is that case.

**Teeth, pinned and not merely demonstrated.**
`test_the_structural_walk_has_teeth` runs eight synthetic sources through both
scanners and asserts each one's premise as well as its verdict:

```
same line (regex CAN see)    regex 1  |  AST walk 1
CONTINUATION LINE            regex 0  |  AST walk 1   <-- caught ONLY by the new walk
non-zero sentinel 1.0+0.0j   regex 0  |  AST walk 0
.astype recovery             regex 0  |  AST walk 0
the fixed spelling           regex 0  |  AST walk 0
both branches literal        regex 0  |  AST walk 0
literal inside a comment     regex 0  |  AST walk 0
real zero fill               regex 0  |  AST walk 0
```

The file goes **239 -> 247 ids, all passing**.

**One genuine new site, in a file I do not own: `elements/doe.py:539`.**
`T = np.where(inside, T, 0.0 + 0j)` -- the regex never saw it because the fill
is spelled `0j`, not `0.0j`.  Rated **P3 by measurement, not by assumption**:
on that branch `T = np.exp(1j * np.where(is_even, 0.0, np.pi))`, and the phase
is built from Python float literals, so it is float64 for every input the entry
point accepts.  `T` is complex128 unconditionally -- measured complex128 on all
four `(binary, n_zones)` combinations -- and the fill is complex128 too, so
nothing is promoted on any reachable call.  It is added to the pin's own
`_P3_ALLOWLIST` with that measurement written at the entry, which is the
mechanism that file documents for exactly this severity gradient, and it is
also a one-line request to that module's owner (sec. F5).  It is NOT a silent
exemption: the entry says what the migration is and to delete the entry when it
lands.

### F2.7 (F-g) The A17 falsifiability arm was RED, and it was my file

Found while re-running the gate:

```
FAILED ...::test_the_fingerprints_are_actually_sensitive[lumenairy._context]
E   AssertionError: no small integer literal found to re-spell
```

Not a fingerprint problem and not `_context.py`'s fault: the mutation catalogue
ran out of targets.  MEASURED, `lumenairy/_context.py` contains **no integer
constant at all** (0 of them) and 31 single-line string constants.  Mutation 3
needs a literal whose re-spelling preserves the AST and moves the token stream,
and an integer is not the only one that does.

The arm now falls back to flipping a string's quote style (`'a'` -> `"a"`), and
the fallback is careful about the one thing that would make it vacuous: it skips
any string that is a *statement* (docstring or string-as-comment), because the
token fingerprint drops those by design.  It also skips strings containing a
backslash or the other quote character, where the flip would not be
value-preserving.  The "no target at all" message now says it is a gap in the
catalogue and not evidence about the module -- "widen the catalogue rather than
exempting the module".

**116 passed** on that arm alone; **708 passed, 0 failed** for the whole A17
file, the first time this campaign it has been fully green.

## F3. Files touched in the follow-up

**Modified -- library (7)**

* `lumenairy/elements/_lens_thin.py` -- the cycle break (imports,
  `__getattr__`, six call sites, five comments).
* `lumenairy/elements/lenses.py` -- one docstring sentence at `:146`
  (coordinator-authorised; docstring-only, so neither fingerprint moved).
* `lumenairy/elements/pmm/stack2d_pure.py` -- `stacklevel=4` + why-comment.
* `lumenairy/elements/eme/eme_diffraction.py` -- refusal message reworded.
* `lumenairy/elements/pmm/stack.py` -- refusal message reworded.
* `lumenairy/elements/pmm/_core.py` -- `_ARCHIVE_SLANT_FOLD` deleted, section
  header and two pointer comments repointed.
* `lumenairy/propagators/asymptotic_jax_twin.py` -- two dtype-matched fills.

**Modified -- tests (3)**

* `tests/unit/test_audit2609_a21_pmm_warning_filter.py` -- one new fail-before
  test (6 -> 7 ids).
* `tests/unit/test_audit2609_a17_history_relocation.py` -- mutation 3 widened,
  plus a `_node_source` helper and the docstring that explains the fallback.
* `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` -- the structural
  walk, the value-correct literal rule, one measured allowlist entry, and the
  eight-case teeth test (239 -> 247 ids).

**Modified -- history documents (6)** (headers re-recorded; three also gained a
section)

* `docs/history/lumenairy.elements._lens_thin.md` (header only)
* `docs/history/lumenairy.elements.pmm.stack2d_pure.md` (header only)
* `docs/history/lumenairy.propagators.asymptotic_jax_twin.md` (header only)
* `docs/history/lumenairy.elements.eme.eme_diffraction.md` (+ `L167-169`)
* `docs/history/lumenairy.elements.pmm.stack.md` (+ `L3209-3215`)
* `docs/history/lumenairy.elements.pmm._core.md` (+ `L7516-7565`)

**Modified -- documentation (1)**

* `README.md:570` -- the v4.13.2 thin-lens note cited `_lenses_module.cp`, a
  name F2.1 deleted.  Caught by `scripts/check_doc_identifiers.py` on the
  final sweep (denominator 597, 1 unresolved), which is the gate working on
  its first real regression: a code change made a document stale and the
  resolver named the file, line and token.  Reworded to name the accessor the
  sites call today, with the v4.13.2 mechanism kept in a parenthetical so the
  historical note still reads true.  Denominator unchanged at 597, unresolved
  back to 0.

**Modified -- reports (2)**

* `docs/audits/.../fixes/WP-A22_REPORT.md` (this section)
* `docs/audits/.../fixes/WP-A22_CHANGELOG.md` (a follow-up group)

**NOT touched:** `validation/probe_ci_kernel_sweep/*`, `lumenairy/__init__.py`,
`lumenairy/backend/__init__.py`, `lumenairy/elements/lens_config.py`,
`tests/unit/test_ci_kernel_consistency.py` and
`tests/unit/test_audit2609_a23_census_mechanism.py` (all another work
package's, in flight as this one closed -- between them they appear to be
taking sec. 5.3, sec. 5.4 and sec. 5.5), and `lumenairy/elements/doe.py`
(sec. F5).

**One shared file, flagged for the committer.**  `lumenairy/elements/lenses.py`
carries BOTH my three-line docstring correction at `:146` (F2.1) and that other
package's much larger re-export work from `:1040` onward.  Verified at close:
my hunk is intact (`@@ -146,2 +146,5 @@`) and disjoint from theirs, so the two
can be committed together or separately without conflict -- but the diff is not
all mine.

## F4. Tests run in the follow-up

| command | result |
|---|---|
| `test_audit2609_a17_history_relocation.py` + `test_audit2609_a22_history_fingerprint_tool.py` | **708 passed**, 33.1 s |
| `test_audit2609_a17_history_relocation.py::test_the_fingerprints_are_actually_sensitive` | **116 passed**, 23.3 s |
| `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` | **247 passed**, 3.5 s (was 1 failed / 238 passed) |
| `test_audit2609_a12_pmm1d.py`, `test_audit2609_a12_verify_pmm1d.py`, `test_audit2609_a13_stack2d.py`, `test_audit2609_a13_staggered_cost.py`, `test_audit2609_a13_twod.py`, `test_audit2609_a13_verify_guards.py` | **105 passed**, 84.4 s |
| `test_eme_diffraction.py`, `test_eme_2d.py`, `test_eme_2d_vector.py`, `test_audit_w6_eme.py`, `test_niche_audit_w6_eme.py`, `test_fix_eme_branch_cut.py`, `test_eme_census_determinacy.py`, `test_audit2609_a14_rcwa_eme_bor.py`, `test_audit2609_a8_thin_elements.py`, `test_audit2609_a8_verify.py`, `test_audit2609_a21_pmm_warning_filter.py` | **271 passed**, 832 s |
| `test_eme_2d.py`, `test_eme_2d_vector.py`, `test_eme_census_determinacy.py`, `test_fix_eme_branch_cut.py`, `test_audit2609_a14_rcwa_eme_bor.py` | **73 passed**, 625 s |
| `test_audit2609_a4_asymptotic.py`, `test_audit_w6_propagators.py`, `test_niche_audit_w6_asymptotic.py`, `test_v5_4_7_walker_v20_cross_backend_parity.py`, `test_v5_1_0_agent_d_split.py` | **169 passed, 1 failed**, 176 s -- the failure is pre-existing, see below |
| `ruff check lumenairy tests scripts` | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **OK**, every document matches |
| `python scripts/check_doc_identifiers.py` | **OK**, denominator 597, unresolved 0 (after the `README.md` fix in F3) |
| FINAL consolidated: `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`, `test_audit2609_a17_history_relocation.py`, `test_audit2609_a22_history_fingerprint_tool.py`, `test_audit2609_a21_pmm_warning_filter.py`, `test_audit2609_a8_thin_elements.py`, `test_audit2609_a8_verify.py`, `test_audit2609_a21_doc_identifiers.py` | **1 064 passed**, 55.4 s |

### One pre-existing failure found, NOT mine

`test_v5_4_7_walker_v20_cross_backend_parity.py::test_jax_intersect_direction_aware_root_pick_present`:

```
E   AssertionError: V20: _intersect_jax_param reintroduced the direction-blind selector.
E   't1 if R' is contained here: ... 'R_finite > 0' is contained here:
E       1).  An ``R_finite > 0`` selector
```

It is a PROSE walker: it calls `inspect.getsource(_intersect_jax_param)` in
`lumenairy/raytrace/jax_trace.py` and greps that text for two phrases, and both
phrases are matching **inside a comment** that describes the selector the guard
forbids -- the classic failure mode of asserting on source text rather than on
structure.

Attribution, structural rather than argued: `lumenairy/raytrace/jax_trace.py`
and the walker file are both CLEAN in `git status`, and I touched neither, so
the bytes that assertion reads are byte-identical to `HEAD` -- the failure
reproduces at `HEAD` by construction.  It is also not a sweep-3 regression: the
phrase has been inside that function's body both before (`2ede9a16~1`, line
1565) and after (`2ede9a16`, line 1557) that commit, and `git log -S` dates it
to `dc50995e` (v3.5.6).

Owner: whoever holds `raytrace/jax_trace.py` / that walker.  The fix is to
assert structurally (walk the AST for the selector expression) rather than to
re-word the comment, which is what a text walker will keep demanding.

**No new failures.**  The pre-existing failures recorded in sec. 4 are
unchanged in status: sec. 4(a) `test_ci_kernel_consistency.py` still needs the
census owner (sec. 5.5), and sec. 4(c) `mypy` still needs the two annotations in
`lumenairy/backend/__init__.py` (sec. 5.3).  Sec. 4(b) -- the 46 transient A17
failures on mid-write documents -- is **resolved**: that file is now fully green.

## F5. Requests still open

* **sec. 5.3** `lumenairy/backend/__init__.py`, two annotations --
  MERGE-BLOCKING, still open.
* **sec. 5.4** the 31 re-exports, to whitelist the root `__init__` -- open.
* **sec. 5.5** the stale CI-kernel census -- open (and another agent is in
  `validation/probe_ci_kernel_sweep/` right now, which may be exactly this).
* **NEW: `test_v5_4_7_walker_v20_cross_backend_parity.py::test_jax_intersect_direction_aware_root_pick_present`**
  -- a prose walker matching its own forbidden phrases inside a comment in
  `raytrace/jax_trace.py`.  Pre-existing at `HEAD`; detail in sec. F4.
* **NEW: `lumenairy/elements/doe.py:539`** -- one line, for that module's
  owner:

  ```python
  -            T = np.where(inside, T, 0.0 + 0j)
  +            T = np.where(inside, T, np.zeros((), T.dtype))
  ```

  P3 today by the measurement in sec. F2.6 (nothing is promoted on any
  reachable call), allowlisted in the pin with that measurement written out,
  and P1 the moment the phase is built at a narrower dtype.  When it lands,
  delete the `('elements/doe.py', 539)` entry from `_P3_ALLOWLIST` in
  `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` in the same
  change, so the walk confirms it rather than exempting it.

Sections 5.1, 5.2, 5.6 and 5.7 are now **closed** -- 5.1/5.2/5.6 by this
follow-up, 5.7 by the sweeps landing.

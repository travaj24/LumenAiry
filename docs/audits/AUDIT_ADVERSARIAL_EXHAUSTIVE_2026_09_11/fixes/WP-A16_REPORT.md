# WP-A16 -- configuration objects for the `apply_real_lens` family, plus the lens-side architecture and import-time items

Branch `audit-fixes-2026-09`, working tree, no git writes.
All measurements on this workstation (Windows 11, CPython 3.14, NumPy 2.x,
numba present, CuPy ABSENT, `OPENBLAS_NUM_THREADS=1` on every invocation).
The box was shared with several other agents throughout; where that matters to
a number it is said so explicitly.

---

## 1. Summary

| item | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| **V6 / section 14 item 13** -- config objects for the lens family | **done** | new `lumenairy/elements/lens_config.py` (1207 lines; dataclasses at :208/:305/:445/:841, resolver at :1095); `_lens_real.py:162-171,4623-4626,5514-5517`; `_lens_traced.py:44-51,7582-7585,8718-8721,14732-14735,14815-14818`; `lenses_maslov.py:51-58,1464-1467,1686-1689`; `lenses_gbd.py:50-57,279-282,407-410`; `fga.py:98-108,1559-1562,1763-1766`; `_lens_traced_multibranch.py:63-70,521-524,627-632`; `elements/__init__.py:94-99,267-268`; `lumenairy/__init__.py:143-148,1191-1196` | `test_audit2609_a16_lens_config_round_trip.py` (84 ids), `test_audit2609_a16_lens_config_bit_identity.py` (27 ids) | `np.array_equal` against the identical keyword call, on WP-A15a's covering-array fixture | 0 config objects -> 38 fields over 7 entry points; **max \|diff\| = 0 (exactly) on all 8 configured cases x 3 spellings** |
| **Addendum 8** -- five optional-dep sites -> `backend/_optional.py` | **done** | `_lens_real.py:31-69`; `_lens_traced.py:28-117`; `lenses.py:44-73,81-118,139-158`; `_lens_imap.py:406-437`; `fga.py:93-118` | `test_audit2609_a16_lens_arch.py` (27 ids) | AST census of own `import cupy` / `import numba` inside the three helper names | **7 own copies -> 0** |
| **Addendum 9** -- three lens knobs registered | **done** | `_lens_real.py:213-228,1409-1437`; `_lens_traced.py:571-586` | same file | `_knobs.knobs()`, `override()` apply+restore incl. the exception path, `setter(getter())` round trip | **registry 15 -> 18 at `import lumenairy`** (the audit's 20 counting the lazily-imported owners); `set_low_memory` coverage **3/4 -> 4/4** |
| **Addendum 10** -- `_lens_*`/`lenses` 2-cycles | **deferred, plan written** | `docs/lens_configuration.md` section "Module layout" | -- | module-level-only AST walk over the 11 lens modules | 4 cycles at HEAD -> **4 cycles now** (unchanged); `lens_config` adds **0** new edges into the family |
| **Addendum 11** -- exports | **done** | `lumenairy/__init__.py`, `elements/__init__.py` | `test_public_api.py`, `test_v4_16_0_walker_all_symmetry.py` | `__all__` resolvability walkers | `lumenairy.__all__` **+4**, `elements.__all__` **+4** |
| **Addendum 12** -- narrow the named broad except | **done** | `_lens_imap.py:1574-1586` | `test_audit2609_a16_lens_arch.py` | AST census of `except Exception` / bare `except` in the module | **1 -> 0** broad clauses in `_lens_imap.py` |
| **Addendum 12** -- the two F541 | **done** | `_lens_real.py:2701-2702` | `test_audit2609_a16_lens_arch.py` | `ruff --isolated --select F541` on the HEAD blob vs the working tree | **2 -> 0** |
| **Addendum 14** -- lazy `backend.scipy` + the `airy` import | **done** | `backend/__init__.py:30,56-81`; `_lens_traced_multibranch.py:74-97,215-222,241` | `test_audit2609_a16_lens_arch.py` (fresh-interpreter `sys.modules` check + the Airy oracle) | interleaved same-build A/B of `perf_counter` around `import lumenairy`; `sys.modules` membership | `scipy.linalg` in `sys.modules` after `import lumenairy`: **True -> False**; `import lumenairy` **745.0 -> 673.8 ms** (medians of 9, quiet box); interleaved same-build A/B delta **+0.1 ms -> +70.8 ms** |

No finding was found not-reproducible.  One item (addendum 10) is deferred with
a concrete plan and a two-line first step that belongs to a file I do not own.

---

## 2. Per deliverable

### 2.1 The config objects (the main deliverable)

**What the audit asked for and why.**  TESTS-ARCH section 14 item 13 sketches
three frozen dataclasses over the family's ~20 shared parameters and names
three wins: settings become testable in isolation, a covering array over
dataclass fields becomes trivial, and "the 'knob silently discarded' class of
bug becomes a `dataclasses.fields()` round-trip assertion".  The motivating
measurement is in the same report: 673 call sites, 177 distinct combinations,
68 % passing zero or one optional keyword, and four of five seeded defects in
the *interaction* class the suite had no power against.

**What I read before writing it.**  The live signatures of all seven entry
points (dumped by `inspect.signature`, not transcribed): `apply_real_lens` 31
keyword-only parameters, `apply_real_lens_traced` 51,
`prepare_real_lens_traced` 28, `apply_real_lens_maslov` 32,
`apply_real_lens_gbd` 29, `apply_real_lens_fga` 24,
`apply_real_lens_traced_multibranch` 14 (counts as of this change, i.e.
including the four it adds).  Plus `WP-A2/A3/A4_REPORT.md` and their
`VERIFY_*` files for what changed in those signatures days ago -- the two main
entry points are keyword-only after `E_in` (WP-A15a section 2.1) and the
covering-array fixture factory is importable on purpose.

**The partition.**  By ROLE, and documented field by field in
`docs/lens_configuration.md`:

* `LensGeometry` (10) -- what optical problem is being solved;
* `LensNumerics` (17) -- how that same problem is discretised;
* `LensResources` (11) -- what machine may be used and what is reported.

38 fields over ~110 distinct parameters.  Everything else stays keyword-only,
each with a written reason in `lens_config.KWARG_ONLY` and in the docs page,
in four classes: physics-model flags (a fourth role -- see "Deferred"),
single-model tuning constants, private mutable out-parameter sinks, and names
whose DEFAULTS disagree between siblings.

**Three cross-entry-point naming defects found while deriving the tables.**
None was changed (renaming a shipped keyword is a migration, not a refactor),
all three are now documented and encoded:

1. `apply_real_lens_traced_multibranch.ray_subsample` (default **2**) is NOT
   `apply_real_lens_traced.ray_subsample` (default **8**).  The multibranch one
   is the CAUSTIC launch spacing, i.e. the sibling of the traced engine's
   `caustic_ray_subsample` (also 2).  A user moving `ray_subsample=8` between
   the two would silently change the caustic launch density by 4x.  The config
   field is `caustic_ray_subsample` and the table renames it for multibranch.
2. `multibranch.min_area_ratio` is `traced.caustic_min_area_ratio` -- same
   setting, same default 1e-6, two spellings.  Renamed in the table.
3. `multibranch.input_carrier` is NOT `carrier`.  It is a transverse carrier
   WAVEVECTOR (`None | 'auto' | (kx, ky)` in rad/m); `carrier=` on the analytic
   and traced entry points is a reference CONGRUENCE (`None | 'auto' |
   conjugate distance in m | wavefront ndarray | TiltedCarrier`).  Mapping them
   onto one field would silently reinterpret metres as rad/m, so
   `input_carrier` is keyword-only with that reason recorded.

Plus two DEFAULT CLASHES that force a keyword-only classification:
`normalize_output` is `'power'` on maslov and `'none'` on gbd/fga;
`mem_budget_mb` is `512.0` on gbd and `None` on fga.  And the work-chunk size
is spelled three ways with three defaults (`chunk_v2=64`,
`chunk_beamlets=2048`, `chunk=None`).

**How the entry points take it.**  Each gained
`geometry=/numerics=/resources=/config=`, all `None`, at the END of the
keyword-only list, and a three-line block immediately AFTER the
`_check_2d_scalar_field` guard (which must stay the first executable statement
-- `test_v4_15_3_dispatcher_pin_2d_scalar_field.py` walks for it):

```python
    if _wants_config(geometry, numerics, resources, config):
        return apply_real_lens(E_in, **_resolve_lens_config(
            apply_real_lens, locals(), geometry=geometry, numerics=numerics,
            resources=resources, config=config))
```

The resolver merges the requests into the keywords and the entry point is
RE-ENTERED with them.  That is the design decision that makes bit-identity
structural rather than reviewed: the configured path *is* the keyword path,
one frame deeper, and the merged mapping has the four config parameters
removed so the recursion terminates.  Cost when nothing is configured: four
`is not None` tests.  On `apply_real_lens_traced` the block sits BEFORE the
`caustic='wave'` `locals()` snapshot, so that snapshot sees resolved keywords
and four `None`s, which the wave recursion then carries harmlessly.

**The precedence rule**, as implemented and as documented:

* a FIELD is a request iff it differs from its dataclass default;
* a KEYWORD is a request iff it differs from its signature default;
* two requests that disagree -> `ValueError` with the section 2 prefix;
* exactly one request -> that value; neither -> the entry point's own default.

The two defaults are equal by construction and a test asserts it for every
(entry point, field) pair -- which is what makes the comparison well defined
and what would catch a sibling's default drifting.

A field that is SET but that the entry point has no parameter for RAISES,
naming the entry points that do take it and pointing at `narrowed_to`.  That
was a deliberate choice over warn-and-continue: silent discard is precisely the
failure class the audit found, and `narrowed_to(entry_point)` is the one-line
explicit "yes, drop those" for the cross-engine case.

Documented consequence, pinned by its own test: a field left AT its default is
not a request and therefore cannot conflict -- writing
`LensNumerics(bandlimit=True)` next to `bandlimit=False` gives `False`.  The
alternative (recording which fields were passed to `__init__`) makes
`dataclasses.replace` and unpickling ambiguous for no real gain, and it is what
makes an all-default config free.

**Validation.**  `__post_init__` carries the field-local checks and BORROWS the
entry point's own vocabularies rather than restating them -- `_VALID_SURFACE_MODELS`,
`_VALID_WAVE_PROPAGATORS`, `_VALID_REMAP_ORDERS`, `_VALID_ACCUMULATOR_STORE`
are read live from `_lens_real` through a cached in-function import (module
scope would be a cycle, since `_lens_real` imports `lens_config`).  The enum
checks that live INLINE inside the 5 700-line bodies (`newton_fit`,
`inversion_method`, `amplitude_model`, `caustic`, `remap_sampling`) are NOT
duplicated: the entry point still raises, and the docs page says so.  Likewise
every cross-field and prescription-dependent rule stays where it is.

One place the config is deliberately STRICTER than the keyword: `sag_dtype`.
The keyword path resolves an unrecognised dtype silently to float64
(`_resolve_sag_real`); the config refuses anything but float32/float64, with
the same rule `set_lens_sag_dtype` enforces.  A config object that accepted
`np.float16` and quietly gave float64 would be the discarded-setting failure
this module exists to close.  Recorded in the docs page.

**`LensConfig` API.**  `from_kwargs(**kw)` / `to_kwargs()` (round trip;
`to_kwargs` emits only requests, so `from_kwargs(**to_kwargs()) == cfg`), both
taking `entry_point=` so they speak that entry point's own spelling (which is
where the two renames live); `to_kwargs(include_defaults=True)` emits all 38;
`narrowed_to(entry_point)`; `requests()`; `field_names()`.  An explicitly
passed component REPLACES the config's, whole -- component granularity, so
`config=` and `numerics=` cannot half-disagree about one field.

**Bit-identity evidence.**  Eight configured cases across the seven entry
points, on WP-A15a's covering-array fixture (curved-rear AC254-ish cemented
doublet, diverging spherical wave from -120 mm, N = 64, dx = 1.125e-4 m).  For
each: the keyword call, the `config=` call and the
`geometry=/numerics=/resources=` call, compared with `np.array_equal` (NOT a
tolerance -- a re-dispatch has no numerical content, so any difference is a
defect and a tolerance would only hide one):

```
  ok    apply_real_lens                     n=4096  |E|sum=7.537164514e+02  [dy, surface_model='displaced', conjugate, wave_propagator='asm', remap_order, bandlimit, sag_chunk_rows, accumulator_store, use_gpu]
  ok    apply_real_lens                     n=4096  |E|sum=7.538150566e+02  [wave_propagator='rs', sag_dtype=float32]
  ok    apply_real_lens_traced              n=4096  |E|sum=3.421634691e+02  [newton_fit='spline', newton_poly_order=4, newton_max_iters=40, amplitude_model='ray_density', caustic='multibranch', output_plane_distance, caustic_band='plain', caustic_ray_subsample, fit_radius_beam_factor, parallel_amp, n_workers]
  ok    apply_real_lens_traced              n=4096  |E|sum=7.526242592e+02  [inversion_method, inverse_map, sag_chunk_rows, origin, min_coarse_samples_per_aperture]
  ok    apply_real_lens_maslov              n=4096  |E|sum=6.470758422e+02  [dy, output_plane_distance, output_plane_n, output_subsample, use_gpu, verbose]
  ok    apply_real_lens_gbd                 n=4096  |E|sum=3.246375386e+02  [dy, output_plane_distance, output_plane_n, clip_aperture, output_subsample, verbose]
  ok    apply_real_lens_fga                 n=4096  |E|sum=2.678941725e+04  [dy, output_plane_distance]
  ok    apply_real_lens_traced_multibranch  n=4096  |E|sum=3.845303691e+02  [output_plane_distance, output_plane_n, ray_subsample, min_area_ratio, caustic_band='plain']

FAILS: 0
prepare_real_lens_traced screen bit-identical: True
```

Every case is `max |diff| = 0` exactly; the `|E|sum` column is printed so a
future reader can see the eight arms really produce eight different fields.
`test_an_all_default_config_is_the_unconfigured_call` adds the other half:
`LensConfig()` reproduces the no-config call byte for byte on all eight.

**The round-trip gates are not vacuous** -- six in-process corruptions, each
restored afterwards, nothing written to disk:

```
FAIL-BEFORE C1 (a signature default drifts from the field default)
  C1 ok (AssertionError) -> apply_real_lens_traced_multibranch: the lens_config field tables disagree with the signature: | LensNumerics.caustic_band defaults to 'ludwig' but apply_real_lens_trace...
FAIL-BEFORE C2 (a new keyword nobody classified)
  C2 ok (AssertionError) -> apply_real_lens: ['stream_transfer_function'] are keyword parameters that lens_config neither carries as a field nor documents as deliberately keyword-only.
FAIL-BEFORE C3 (a stale exclusion)
  C3 ok (AssertionError) -> apply_real_lens: KWARG_ONLY documents ['a_keyword_that_was_deleted'], which apply_real_lens no longer accepts.  Delete the entry -- a stale exclusion is a hiding place.
FAIL-BEFORE C4 (silent discard instead of a refusal)
  C4 ok (Failed) -> DID NOT RAISE ValueError
FAIL-BEFORE C5 (a keyword and a field disagree and nothing raises)
  C5 ok (AssertionError) -> the refusal does not name the setting and say that the two spellings must agree
FAIL-BEFORE C6 (the resolver ignores the config entirely)
  C6 ok (AssertionError) -> apply_real_lens [config=]: the configured call differs from the identical keyword call by max |diff| = 5.476732e-04.
RESTORED CHECK
  config == keyword again: True
  refusal is back: apply_real_lens: numerics.newton_poly_order=8 is not a setting apply_real_lens accepts (it ...
```

C4/C5/C6 replace the resolver with the pre-config-object behaviour (take what
fits, drop the rest) and with a resolver that ignores the config outright;
C1/C2/C3 corrupt the tables the way a future signature change would.

**Residual risk.**  (a) The re-dispatch adds one stack frame, so a
`warnings.warn(..., stacklevel=N)` inside an entry point points one frame too
shallow for a caller who used a config object.  Only the configured path is
affected and the message text is unchanged; fixing it properly means threading
a stacklevel offset through ~40 warn sites, which is not worth it for a
cosmetic attribution.  Recorded, not fixed.  (b) `_signature_info` caches by
`id(fn)` with the function held in the entry; an `importlib.reload` of an entry
point's module rebinds `fn` and the guard (`hit['fn'] is fn`) forces a refresh,
but a reload that recycles the id of a DEAD function object would be served a
stale entry.  That is the same exposure `functools.lru_cache` on a function
argument has, and no test in this repo reloads a lens module.

---

### 2.2 Addendum 8 -- the five optional-dependency sites

**FAIL-BEFORE (AST census of the committed HEAD blobs, `git show`, nothing
checked out):**

```
FAIL-BEFORE 1 (optional-dep census) ->
    lumenairy/elements/_lens_real.py:36 _ensure_cupy_loaded imports cupy
    lumenairy/elements/_lens_traced.py:40 _ensure_cupy_loaded imports cupy
    lumenairy/elements/_lens_traced.py:76 _load_numba imports numba
    lumenairy/elements/lenses.py:48 _ensure_cupy_loaded imports cupy
    lumenairy/elements/lenses.py:87 _load_numba imports numba
    lumenairy/elements/_lens_imap.py:417 _load_numba imports numba
    lumenairy/propagators/fga.py:97 _load_numba imports numba
    -> 7 own copies at HEAD (test allows 0)
FAIL-BEFORE 1b (shared CUPY_AVAILABLE) ->
    _lens_real.py: own find_spec probe = True, imports shared constant = False
    _lens_traced.py: own find_spec probe = True, imports shared constant = False
    lenses.py: own find_spec probe = True, imports shared constant = False
```

**What I changed**, exactly per WP-A15b section 5.4 and copying `fft_infra.py`'s
wrapper shape:

| file | before | after |
|---|---|---|
| `_lens_real.py:31-69` | own `CUPY_AVAILABLE`, `_ensure_cupy_loaded`, `_is_cupy_array` | `from ..backend._optional import CUPY_AVAILABLE, ensure_cupy, is_cupy_array` + two thin wrappers keeping the module `cp` alias |
| `_lens_traced.py:28-117` | same three + `_NUMBA_AVAILABLE`/`_load_numba` | same, plus `_NUMBA_AVAILABLE = _OPTIONAL_NUMBA_AVAILABLE` kept a module ATTRIBUTE and `_load_numba` delegating to `numba_handles()` |
| `lenses.py:44-73,81-118,139-158` | same three + `_NUMBA_AVAILABLE`/`_load_numba` | same |
| `_lens_imap.py:406-437` | `_NUMBA_AVAILABLE`/`_load_numba` | same |
| `fga.py:93-118` | `_NUMBA` + `try: import numba` | `load_numba()`, module memo kept, **the `ImportError` at `fga.py:399` untouched** |

Three behaviours preserved, each with its own test:

* **the `cp` alias** -- `_is_cupy_array(x) is True` must imply the module-level
  `cp` is bound, because the GPU branches read the NAME (`xp = cp if
  _is_cupy_array(E) else np`).  The wrapper calls `_ensure_cupy_loaded()` on a
  True answer, and `test_cupy_probe_is_the_shared_one_and_keeps_the_module_cp_alias`
  pins the coupling structurally.  It also pins that the module no longer
  contains its own `find_spec('cupy')` -- without that half, the value check
  is vacuous on a CuPy-free box (`False is False`).
* **`_NUMBA_AVAILABLE` stays monkeypatchable** --
  `test_numba_gate_stays_a_module_attribute_that_monkeypatch_reaches`
  monkeypatches it to `False` on each of the three consumers and asserts
  `_load_numba()` returns `False`.  Without that, every existing test that
  fakes an absent accelerator on these modules would have gone silently
  vacuous.
* **fga's raise** -- `test_fga_keeps_its_deliberate_importerror_for_the_missing_accelerator`
  asserts the `ImportError` still names the function and carries the
  `pip install lumenairy[numba]` hint.

CuPy is NOT installed here, so the CUDA arm is desk-checked only; that is
stated in the test file and here.  The NumPy-2.x `ndarray.device` trap the
deleted copies' comments cite is re-measured every run
(`test_the_numpy_two_device_trap_is_live_so_the_isinstance_test_matters`), so
the isinstance requirement cannot quietly stop being load-bearing.

`_lens_thin.py` also delegates into `lenses` for the CuPy answer; it is not in
my ownership and is listed in section 5.

---

### 2.3 Addendum 9 -- the three lens knobs

**FAIL-BEFORE:** `register_knob` appears nowhere in either file at HEAD (all
three probes False; transcript in section 4).  Before the change,
`lumenairy.override(lens_sag_dtype=np.float32)` raised
`ValueError: override: unknown knob 'lens_sag_dtype'`.

Registered exactly per WP-A15b section 2.2's contract, one call beside each
`set_*`/`get_*` pair:

| knob | getter | setter | owner |
|---|---|---|---|
| `lens_sag_dtype` | `get_lens_sag_dtype` | `set_lens_sag_dtype` | `_lens_real.py:213-228` |
| `pointwise_cos_grid_cache_budget` | `_get_pointwise_cos_grid_cache_budget_bytes` | `_set_pointwise_cos_grid_cache_budget_bytes` | `_lens_real.py:1409-1437` |
| `lens_parallel_amp` | `get_lens_parallel_amp` | `set_lens_parallel_amp` | `_lens_traced.py:571-586` |

**The third needed a private pair, for a real reason.**  The PUBLIC
`set_pointwise_cos_grid_cache_budget(mb)` takes MEGABYTES while
`get_pointwise_cos_grid_cache_budget()` returns BYTES, so `setter(getter())` --
the property the whole snapshot/restore machinery rests on -- would have
multiplied the budget by 2**20 on every restore.  The registry uses a
byte-in/byte-out private pair straight onto `ByteBudgetedLRU.set_budget` /
`.max_bytes`, which is exact (and `None`, "bound only by the global budget",
survives as `None`).  `test_the_megabyte_byte_asymmetry_that_forced_a_private_pair_is_real`
re-measures the asymmetry every run, so if the public pair is ever made
symmetric the private accessors can be retired and that test says so.

Both getters are one global read / one attribute read, side-effect-free, as
the contract requires.

MEASURED: registry **15 -> 18** knobs at plain `import lumenairy` (A15b's 17
includes three whose owner modules -- `io.storage`, `user_library`,
`elements.rcwa` -- are lazily imported).  `override()` apply + restore + the
exception path is pinned for each of the three, and
`test_set_low_memory_is_now_fully_covered_by_snapshot_restore` pins the
downstream consequence: the macro flips four knobs and all four are now
restorable (it was 3 of 4).

---

### 2.4 Addendum 10 -- the module 2-cycles: DEFERRED with a plan

Re-measured with a module-level-only AST walk over the 11 lens modules; it
reproduces the audit's list exactly:

```
2-cycles:
  _lens_real <-> lenses
  _lens_traced <-> lenses
  _lens_thin <-> lenses
  lenses <-> lenses_maslov
```

and the back-edges are completely enumerated (this is new -- the audit counted
them but did not list what they carry):

| back-edge | names imported from `lenses` |
|---|---|
| `_lens_real:146` | `_warn_if_aperture_exceeds_grid`, `surface_sag_biconic`, `surface_sag_general` |
| `_lens_traced:608` | `_warn_if_aperture_exceeds_grid` |
| `_lens_thin:39` | `CUPY_AVAILABLE` (+ a module alias for the lazy `cp`) |
| `lenses_maslov:118` | `NUMEXPR_AVAILABLE`, `_ensure_numexpr_loaded`, `_fit_normaliser`, `_multi_indices_total_degree`, `_warn_if_aperture_exceeds_grid` |

The brief says to extract `elements/_lens_kernels.py` ONLY if bit-identity can
be proved, else to document the plan.  I did not extract it, for two concrete
reasons:

1. **It cannot be proved bit-identical by relocation alone.**
   `surface_sag_general` and `surface_sag_biconic` are several hundred lines
   each with accumulated guards; moving them is a large diff whose only
   defensible gate is the `-k real_lens` slice plus the WP-A2/A3/A4 files, and
   the audit itself budgets **3 days** for the extraction.
2. **It collides head-on with WP-A17 part 2**, which is relocating the
   `v<N>.<N> (audit ...)` narrative blocks out of exactly these files
   (-10 500 lines by the audit's measurement).  Two large mechanical diffs over
   the same regions in the same round is how a relocation loses a guard.

The plan is written up in `docs/lens_configuration.md` section "Module layout",
including the one cycle that is **already a two-line fix**: `_lens_thin` needs
only `CUPY_AVAILABLE` and the lazy `cp`, both of which now live in
`backend/_optional.py`.  That file is not mine -- request in section 5.

`lens_config.py` is deliberately a LEAF (no `lumenairy` imports at module
scope; its validators borrow `_lens_real`'s vocabulary tuples through a cached
in-function import), so adding it created **no new edge into the family** --
verified by the same walk.

---

### 2.5 Addendum 12 -- the narrowed except and the two F541

**Broad except.**  `_lens_imap.build_inverse_map`'s RAM-budget diagnostic wrapped
a bare `from .. import memory` in `except Exception: # noqa: BLE001`.  The only
thing that can fail there is the import itself (the `getattr` has a default),
so it is now `except ImportError:` with a why-comment.  FAIL-BEFORE, from the
HEAD blob:

```
FAIL-BEFORE 4 (_lens_imap broad except) ->
    'except Exception:                                  # noqa: BLE001'
    -> 1 broad except clauses at HEAD (test allows 0): [1570]
```

`test_no_broad_except_remains_at_that_site` walks the whole module's AST and
asserts zero, which is stricter than the `noqa` removal and matches what the
per-file census in `test_audit_except_budget.py` will allow once WP-A15a's
entry is lowered (request in section 5).

**F541.**  MEASURED with ruff itself rather than a heuristic -- which matters,
because F541 is a property of the whole implicitly-concatenated literal, not of
a line: the two `f"  DROPPED: ..."` parts a few lines above are LEGAL (each is
concatenated with a part carrying `{curved}` / `{sorted(set(shifted))}`) and I
left them alone, so the diff is exactly the finding.

```
FAIL-BEFORE 5 (F541 in the mirror guard) ->
    HEAD blob: 2 F541
        _lens_real.py:2644:11: F541 [*] f-string without any placeholders
        _lens_real.py:2645:11: F541 [*] f-string without any placeholders
    working tree: 0 F541
    HEAD spelling of the pair:
        + f"  Use lumenairy.io.split_prescription_at_mirrors(rx) with " | f"apply_mirror at each fold to carry them.
```

The warning TEXT is unchanged; the test also fires the warning on a
curved-mirror prescription and asserts it still interpolates
`CURVED mirror(s) at [1]`, so dropping a prefix cannot take a placeholder with
it unnoticed.

---

### 2.6 Addendum 14 -- import time

**What I changed.**  `lumenairy/backend/__init__.py` replaces
`from . import scipy as scipy` with the PEP 562 forward WP-A15b section 5.1
specified (plus `__dir__`), and `_lens_traced_multibranch.py`'s module-level
`from scipy.special import airy as _scipy_airy` became a cached `_airy()`
accessor with the single call site updated.  `ludwig_fold` is fully
elementwise, so `_airy` runs ONCE per fold, not once per pixel -- the docstring
that used to justify the module-level import is corrected accordingly.

**FAIL-BEFORE (HEAD blobs):**

```
FAIL-BEFORE 3 (backend/__init__ eager scipy) ->
    "from . import scipy as scipy" at HEAD = True
    PEP 562 __getattr__ at HEAD = False
    module-level "from scipy.special import airy" at HEAD = True
    before touch: False        <- after the change: import lumenairy leaves scipy.linalg unloaded
    after touch : True         <- ...and touching la.backend.scipy loads it, so the gate is not vacuous
```

**MEASURED import time, A15b's interleaved same-build method.**  Each sample is
a fresh interpreter; the two arms alternate pair by pair so machine load hits
both equally.  Arm *bare* = `import lumenairy`; arm *eager* = `import
lumenairy` plus an explicit import of exactly what this change made lazy
(`lumenairy.backend.scipy` and `scipy.special.airy`).  Medians of 9 pairs:

| tree / pass | arm | median (ms) | the 9 samples |
|---|---|---|---|
| BEFORE (HEAD) | bare | **745.0** | 686.0 729.2 729.4 742.4 **745.0** 749.6 759.8 761.9 775.5 |
| BEFORE (HEAD) | eager | 745.1 | 672.6 694.8 741.1 743.4 **745.1** 745.7 774.4 801.9 813.6 |
| | | **delta +0.1 ms** | as expected -- both arms were already eager |
| AFTER, pass 1 (LOADED box) | bare | 777.3 | 732.4 736.9 760.7 773.3 **777.3** 785.7 786.8 801.2 868.3 |
| AFTER, pass 1 (LOADED box) | eager | 822.3 | 794.8 797.1 812.9 820.0 **822.3** 832.9 854.8 875.8 1011.4 |
| | | **delta +45.0 ms** | |
| AFTER, pass 2 (quiet box) | bare | **673.8** | 630.5 644.6 665.3 673.7 **673.8** 680.9 681.8 692.3 697.9 |
| AFTER, pass 2 (quiet box) | eager | 744.6 | 696.3 715.2 718.4 718.7 **744.6** 748.2 755.7 756.0 757.4 |
| | | **delta +70.8 ms** | |

**Read the DELTA column first.**  Pass 1 ran while this work package's own
`-k real_lens` and a2/a3/a4 batteries were running, which is why its *bare*
median (777.3 ms) sits ABOVE the BEFORE one even though strictly less is
imported -- exactly the cross-run noise A15b hit ("the 39 ms median difference
is inside that spread").  Pass 2 ran on a quiet box, and there the two
independent numbers agree: the interleaved delta is **+70.8 ms** and the plain
before/after median difference is **745.0 -> 673.8 = -71.2 ms (-9.6 %)**.  The
honest statement is therefore "**worth 45-71 ms depending on machine load, ~71
ms when nothing else is running**", and the load-free structural claim is the
one the test asserts: **`scipy.linalg` is no longer in `sys.modules` after
`import lumenairy`**.

Named cumulative rows, same interleaved method (`python -X importtime`, medians
of 5 pairs).  The *bare* column is this tree; the *eager* column is this tree
plus the two imports the change deferred, i.e. the pre-change state:

| module | bare cum (ms) | eager cum (ms) |
|---|---|---|
| `lumenairy` | 670.5 | 672.7 |
| `lumenairy.backend` | **2.4** | 2.4 |
| `lumenairy.backend.scipy` | *(not imported)* | 60.0 |
| `lumenairy.elements` | 496.3 | 484.4 |
| `lumenairy.propagators.fft_infra` | 446.0 | 429.4 |
| `scipy.fft` | **426.5** | 412.2 |
| `scipy.linalg` | *(not imported)* | 59.6 |
| `scipy.special` | 52.8 | 51.4 |

`lumenairy.backend` costs **2.4 ms** now; A15b measured **567.3 ms** for the
same row before the forward, and `backend.scipy` **566.2 ms**.  That whole cost
did not vanish -- it moved to whoever pulls `scipy.fft`, which is the next row
down.

**Why this is ~71 ms and not the ~540 ms the estimate suggested.**  Traced with
a `sys.meta_path` watcher:

```
--- scipy.special requested; stack:
    lumenairy/__init__.py:29        from .analysis import (
    lumenairy/analysis/__init__.py:45    from .coherence import (
    lumenairy/analysis/coherence.py:27   from ..elements.lenses import apply_real_lens
    ...
    lumenairy/propagators/fft_infra.py:112   import scipy.fft as _scipy_fft
    scipy/fft/__init__.py:91             from ._fftlog import fht, ifht, fhtoffset
    scipy/fft/_fftlog_backend.py:4       from ..special import loggamma, poch
```

`scipy.fft` -- imported at module scope by `propagators/fft_infra.py`, which is
not in my ownership and is being edited by the history-relocation agent right
now -- pulls in `scipy.special` on this SciPy build, and with it the ~370 ms
shared prefix (`scipy._lib._array_api` -> `array_api_compat.numpy` ->
`numpy.f2py` -> `charset_normalizer` -> `numpy.testing`) that A15b's estimate
attributed to `scipy.special` alone.  So the `airy` move cannot save that
prefix while `fft_infra` is eager, and what remains is the marginal
`scipy.linalg` cost -- **59.6 ms cumulative** by the named table, **70.8 ms**
by the end-to-end interleaved A/B (the difference is the `backend.scipy` module
body itself, 60.0 - 59.6 ms, plus the `airy` attribute fetch and measurement
spread), against A15b's 47 ms prediction for that component.  Same order, same
sign, and A15b's own prediction was made with `scipy.special` assumed absent.
The path to the remaining ~426 ms is in section 5 item 4.

**Behaviour preserved.**  `la.backend.scipy.X`, `from lumenairy.backend import
scipy`, `importlib.import_module('lumenairy.backend.scipy')`,
`'scipy' in dir(la.backend)`, `'scipy' in la.backend.__all__`, the caching into
`globals()`, and `AttributeError` (never `ImportError`) for an unknown name --
all pinned in `test_audit2609_a16_lens_arch.py`.  The Airy accessor is pinned
against `scipy.special.airy` itself with `np.array_equal` over 17 points, plus
an end-to-end check that `ludwig_fold` stays finite exactly ON the fold (where
the plain branch sum diverges), which is the property the Airy call carries.

---

## 3. Files touched

**New**

* `lumenairy/elements/lens_config.py`
* `tests/unit/test_audit2609_a16_lens_config_round_trip.py`
* `tests/unit/test_audit2609_a16_lens_config_bit_identity.py`
* `tests/unit/test_audit2609_a16_lens_arch.py`
* `docs/lens_configuration.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A16_REPORT.md` (this file)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A16_CHANGELOG.md`

**Modified** (all within the ownership list)

* `lumenairy/elements/_lens_real.py` -- optional-dep delegation; two knob
  registrations + the private byte accessors; the `lens_config` import; the
  four config parameters + the resolve block + docstring section on
  `apply_real_lens`; two F541 prefixes dropped.
* `lumenairy/elements/_lens_traced.py` -- optional-dep delegation (CuPy +
  numba); `lens_parallel_amp` registration; the `lens_config` import; config
  parameters + resolve block + docstring on `apply_real_lens_traced` and
  `prepare_real_lens_traced`.
* `lumenairy/elements/lenses.py` -- optional-dep delegation (CuPy + numba).
* `lumenairy/elements/_lens_imap.py` -- optional-dep delegation (numba); the
  narrowed `except ImportError`.
* `lumenairy/elements/_lens_traced_multibranch.py` -- the `_airy()` accessor;
  the `lens_config` import; config parameters + resolve block + docstring.
* `lumenairy/elements/lenses_maslov.py` -- the `lens_config` import; config
  parameters + resolve block + docstring.
* `lumenairy/elements/lenses_gbd.py` -- the `lens_config` import; config
  parameters + resolve block + docstring.
* `lumenairy/propagators/fga.py` -- optional-dep delegation (numba); the
  `lens_config` import; config parameters + resolve block + docstring.
* `lumenairy/backend/__init__.py` -- PEP 562 lazy `scipy` + `__dir__`.
* `lumenairy/elements/__init__.py` -- four exports + four `__all__` entries.
* `lumenairy/__init__.py` -- four exports + four `__all__` entries.

**Not touched, though in the ownership list:** `lumenairy/elements/_lens_jax.py`,
`lumenairy/elements/_lens_traced_uniform.py`, `lumenairy/propagators/gbd.py`.
`_lens_jax` and `_lens_traced_uniform` are wrappers over entry points that now
take the config parameters and inherit them for free; `propagators/gbd.py` has
no lens entry point of its own (`lenses_gbd.py` is the entry point and it calls
into `gbd`).  No optional-dep copy, knob or F541 in any of the three.

---

## 4. Tests run

Every command with `OPENBLAS_NUM_THREADS=1`, serial, `-p no:cacheprovider`, on
a box shared with other agents.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a16_lens_config_round_trip.py -q` | **84 passed** | 1.3 s |
| `pytest tests/unit/test_audit2609_a16_lens_config_bit_identity.py -q` | **27 passed** | 8.6 s |
| `pytest tests/unit/test_audit2609_a16_lens_arch.py -q` | **27 passed** | 3.3 s |
| `pytest` over the ten `test_audit2609_a2_*` / `a3_*` / `a4_*` files, `-q` | **300 passed**, 1 warning | 146.4 s |
| `pytest tests/unit -k real_lens -q` | **158 passed, 3 skipped**, 14232 deselected | 265.9 s |
| `pytest` over 17 files (the three a16 files, the a15a covering array, the four a15b files, test_public_api, the all_symmetry / dy_threading / shell_vs_canonical walkers, the two dispatcher pins, niche_audit_e_prepared_and_enums, audit_except_budget, a15a_packaging), `-q` | **394 passed, 12 failed** -- all 12 pre-existing, see below | 43.2 s |
| `pytest tests/unit/test_audit2609_a4_maslov_gbd.py tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py -q` | **69 passed** (the same 12 ids, green, with jax x64 enabled first) | 20.6 s |
| `python validation/run_all.py test_lenses` | **PASS** (1/1 files) | 27.0 s |
| `ruff check lumenairy/ --no-cache` | **All checks passed** | |
| `ruff check tests/unit/test_audit2609_a16_*.py --no-cache` | **All checks passed** | |
| `python -c "import lumenairy"` | OK | |

The three new files are **138 ids in 13.2 s** total -- comfortably inside the
fast lane, no `slow` marker needed.

The `-k real_lens` slice grew from the brief's 135 to **158** because 21 of the
new round-trip file's parametrised ids are named after the entry points
(`...[apply_real_lens]`, `...[apply_real_lens_traced]`, ...) and `-k` matches
ids, plus the two `prepare_real_lens_traced` cases.  Nothing was deselected
that used to be selected.

### Pre-existing failures found, NOT mine

**(a) 12 ids in `tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py`** -- the
`apply_real_lens_traced_jax` / `apply_real_lens_maslov_jax` parametrisations of
`TestL4aGlassAfterMirrorDispatcherPin`, `TestDyNoneAcceptanceDispatcherPin`,
`TestDyNeDxBehaviourDispatcherPin` and `TestDtypePreservationDispatcherPin`:

```
RuntimeError: apply_real_lens_traced_jax: the JAX (differentiable) real-lens path
requires double precision, but jax_enable_x64 is disabled ...
```

This is a test-ISOLATION defect of exactly the class WP-A15a section 2.10 is
about, and it is not mine.  Evidence, all measured:

* the file NEVER enables `jax_enable_x64` itself (grep; 12 other test modules
  do) and `tests/conftest.py` does not either, so it depends on collection
  order;
* run ALONE it fails 12 / passes 23; run after `test_audit2609_a4_maslov_gbd.py`
  (which enables x64) it is **69 passed, 0 failed**;
* in the `-k real_lens` slice, where many x64-enabling modules are collected
  first, those same 12 ids **pass**;
* the raise comes from `lumenairy/elements/_lens_jax.py:67`, and
  `git diff --stat lumenairy/elements/_lens_jax.py tests/conftest.py
  tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py` is **empty** -- I touch
  none of the three.

The fix belongs with the file's owner (an autouse `jax.config.update(
'jax_enable_x64', True)` fixture in that module, or in `tests/conftest.py`
alongside the knob fixture), not in WP-A16.

**(b) `test_v5_2_walker_pep562_forwarding.py::
test_v14_every_mutable_fft_infra_global_is_in_live_forward_names`**

```
Missing entries:
  - _PYFFTW_FIRST_FFT_THREAD  (global X declared @ fft_infra.py:664, ...)
  - _PYFFTW_SHARED_BUFFERS_UNSAFE  (global X declared @ fft_infra.py:664, ...)
```

Both globals come from the UNCOMMITTED `lumenairy/propagators/fft_infra.py`
diff (296 insertions / 473 deletions in the working tree; `git diff` shows them
added), i.e. the history-relocation agent's in-flight work.  I touch neither
`fft_infra.py` nor `propagators/propagation.py`'s forward list.  Request in
section 5 item 5.

### Gates that were RED and are now green because of this work package

`tests/unit/test_audit_except_budget.py` went red mid-change (`elements/
lens_config.py: 3 vs 0` in the per-file census) when the first draft of
`lens_config._same` used three `except Exception:` clauses.  Rather than ask
for a census bump -- the house rule is NARROW > WARN > RE-RAISE > KEEP-AS-IS --
the comparison was restructured into one `try` with
`except (TypeError, ValueError)`, which is the exact set a comparison of two
setting values produces, and the test file now pins both halves: the two
recoverable types return `False`, and a `__eq__` that raises `RuntimeError`
(a bug, not an incomparability) PROPAGATES.  `lens_config.py` carries zero
broad excepts and the census is untouched.

## 5. Requested changes outside my ownership

1. **`pyproject.toml` (orchestrator) -- delete the now-unnecessary per-file
   ignore.**  In `[tool.ruff.lint.per-file-ignores]`, the block WP-A15a added:

   ```toml
   # TODO(audit-2609): _lens_real.py:2644-2645 -- two ``f"..."`` continuation
   # lines in the mirror-guard warning carry no placeholders (F541).  Drop the
   # ``f`` prefix on those two lines.  Owner: the analytic-lens work package.
   "lumenairy/elements/_lens_real.py" = ["F541"]
   ```

   The prefixes are gone (measured: `ruff --isolated --select F541` reports 0
   on the working tree, 2 on the HEAD blob).  Delete the comment and the line.
   **Optionally, in the same edit:** `combine-as-imports = true` under
   `[tool.ruff.lint.isort]` collapses the 31 single-name `X as X` re-export
   statements this work package added to `elements/lenses.py` (section 8.1)
   back into the 8 parenthesised blocks they came from.

2. **`tests/unit/test_audit_except_budget.py` (WP-A15a) -- lower the
   `_lens_imap.py` entry to zero.**  WP-A15a section 2.9 class (5) counts one
   narrowing request there; it is now `except ImportError:` and the module
   carries zero broad clauses.  Both bars are `<=`, so the gate is green either
   way, but the census should come down by one so it does not become slack.
   (`lumenairy/memory.py`'s two are still open, with their own owner.)

3. **`lumenairy/elements/_lens_thin.py` owner -- break the fourth lens cycle in
   two lines.**  *(ACTIONED: the change is in the working tree as of
   2026-09-12; `_lenses_module` is gone and `_lens_thin` now asks
   `backend._optional` directly.  One follow-on, item 9 below.)*  `_lens_thin.py:39` imports `CUPY_AVAILABLE` from `.lenses` and
   `_is_cupy_array` delegates through `_lenses_module`; both now live in
   `lumenairy/backend/_optional.py`, which is a leaf.  Replace:

   ```python
   from .lenses import (..., CUPY_AVAILABLE)
   def _is_cupy_array(x):
       return _lenses_module._is_cupy_array(x)
   ```

   with `from ..backend._optional import CUPY_AVAILABLE, is_cupy_array as _is_cupy_array`
   (keeping the PEP 562 `cp` forward, which can then read
   `backend._optional.cupy_module()` / `ensure_cupy()`).  That removes the
   `_lens_thin <-> lenses` module-level 2-cycle outright, with no code motion.
   A15b section 5.4's last table row anticipated exactly this.

4. **`lumenairy/propagators/fft_infra.py` owner -- the remaining ~426 ms of
   import time.**  *(DONE by WP-A22, commit `7d03d799`: "scipy.fft loaded on
   first use (import lumenairy -53 %)".)*  `fft_infra.py:112`'s module-level `import scipy.fft as
   _scipy_fft` is now the ONLY reason `scipy.special` (and its heavy shared
   prefix) is loaded by `import lumenairy`; measured stack in section 2.6.
   `scipy.fft` costs **426.5 ms cumulative** on this box (measured table in
   section 2.6), against `lumenairy.backend`'s 2.4 ms after my change.  It is
   one of three FFT backends in a priority chain that already defers pyFFTW
   behind `find_spec` + first-use, so the same treatment applies:
   probe with `importlib.util.find_spec('scipy.fft')` at import and bind
   `_scipy_fft` on first use in `_ensure_scipy_fft_loaded()`.  Expected: the
   bulk of the ~540 ms A15b's estimate attributed to this pair.  I did not do
   it: the file is not mine and it is being rewritten right now.

5. **`lumenairy/propagators/propagation.py` owner -- two forward-list entries.**
   *(DONE by WP-A21, commit `949edb3b`; the walker is green -- re-run
   2026-09-12.)*
   `_PYFFTW_FIRST_FFT_THREAD` and `_PYFFTW_SHARED_BUFFERS_UNSAFE` (new mutable
   `fft_infra` globals in the uncommitted diff) need to join the PEP 562 live
   forward names, or `test_v5_2_walker_pep562_forwarding.py::
   test_v14_every_mutable_fft_infra_global_is_in_live_forward_names` stays red.
   See section 4.

6. **Docs agent (README / Migration-Guide) -- one paragraph, if wanted.**  The
   config objects are new public API with a docs page
   (`docs/lens_configuration.md`); a pointer from the README's lens section
   would help people find it.  No migration note is required -- nothing
   changed for an existing caller -- and the CHANGELOG text says so.

7. **`tests/unit/test_niche_d6_exact_tilted_leg.py` owner (WP-A6 / VERIFY-A6) --
   restate `r_on`'s bar with a derivation, or investigate the 0.0268 it lost.**
   `test_decentred_carrier_decentre_penalty_envelope`'s `assert r_on > 0.97`
   reads **0.969787** and has done, bit-stably, since `a18ab074` (WP-A6) -- ten
   commits before WP-A16 and long before this file's bar was last looked at.
   Full bisect, metrics and the reasoning are in section 8.3.  Two possible
   outcomes, and the choice belongs to that subsystem's owner:

   * **If 0.9698 is the post-C1 truth**, restate the bar the way its own
     sibling `r_off` is already written -- two-sided, with the derivation
     TESTING_STANDARDS asks for: the oracle's own EE2 error floor at this
     readout pitch, the measured value, and the decades of gap on both sides.
     A one-sided threshold parked 0.0266 below a single 2026-07-29 measurement
     is a per-build number; it says nothing about what size of defect it
     catches, which is why it broke on a 0.02 % move rather than on a physics
     regression.  Note also that the docstring's premise now reads backwards:
     the DECENTRED ratio (0.9855) is better than the on-axis one (0.9698).
   * **If it is not**, C1 ("focus readout sized from the BEAM") is the first
     thing to look at -- it changes the window the EE2 is counted in, which is
     precisely the failing quantity.

   I did not touch the file: it is not in WP-A16's ownership list, and WP-A16
   is measured not to have moved the number.

8. **`lumenairy/propagators/carrier.py` owner -- the decentre calibration the
   shipped warning quotes is stale.**  `propagate_traced_carrier_chain`'s
   `decentre_fit_frac` `RuntimeWarning` (`carrier.py:8986`) tells users
   "MEASURED ... 0.00 w -> 0.997 ... 1.00 w -> 0.983".  Re-measured on that same
   stand-in today: **0.00 w -> 0.9698, 1.00 w -> 0.9855** -- both ends moved and
   the two have crossed over.  Re-measure the six-point table and restate it, or
   drop the numbers and point at the test that holds them.  Detail in section
   8.4.

9. **`README.md` owner (docs agent) -- one stale identifier, newly created.**
   `README.md:570` names `` `_lenses_module.cp` ``.  The in-flight
   `_lens_thin.py` change (which implements my section 5 item 3) deleted
   `_lenses_module`, so
   `tests/unit/test_audit2609_a21_doc_identifiers.py::test_no_backticked_identifier_in_the_docs_is_unresolved`
   is now red on that one token:

   ```
   _lenses_module.cp  (1x, first at README.md:570)  [no owner exposes the chain]
   ```

   The sentence is a historical note about a fixed CuPy dispatch bug; rewording
   it to name `lumenairy.backend._optional.cupy_module()` (where the lazy `cp`
   now lives) resolves it.  Neither file is mine; flagged because the change
   that caused it is one I requested.

---

## 6. Deferred, with designs

1. **`elements/_lens_kernels.py` and the four lens 2-cycles.**  Design, effort
   and the reasons for deferring are in section 2.4 and, at length, in
   `docs/lens_configuration.md` section "Module layout".  Effort: ~3 d as the
   audit estimates, best taken AFTER WP-A17 part 2's history relocation has
   landed in the same files.  The `_lens_thin` quarter of it is a two-line
   change available today (section 5 item 3).

2. **`LensPhysics`, a fourth dataclass.**  Five `apply_real_lens` keywords --
   `fresnel`, `absorption`, `slant_correction`, `seidel_correction`,
   `surface_frame` (+ `seidel_poly_order`) -- are physics-model options: they
   change which terms the per-surface screen carries.  That is a fourth role
   and the audit's design named three, so rather than force them into the wrong
   one they stay keyword-only, documented as such.  The design is mechanical:
   a fourth frozen dataclass added to `LensConfig` and to `lens_config._GROUPS`
   -- the resolver, `from_kwargs`, `to_kwargs`, `narrowed_to` and every test
   walk over `_GROUPS`, so nothing else changes.  The work is the docstrings
   and the pairwise validation (`slant_correction` and `seidel_correction`
   replace the SAME per-surface coefficient and stacking them double-counts
   the facet obliquity -- WP-A15a measured 173.5 -> 1488.6 nm rms exit OPD on
   an 8 mm cemented doublet; all five are refused under
   `surface_model='displaced'`).  Those are cross-field, so
   `LensPhysics.__post_init__` could carry the pairwise ones while the entry
   point keeps the prescription-dependent ones.  **Effort ~4 h.**

3. **Unifying the three naming mismatches.**  `multibranch.ray_subsample` ->
   `caustic_ray_subsample`, `multibranch.min_area_ratio` ->
   `caustic_min_area_ratio`, and `normalize_output`'s two defaults.  Each is a
   caller-visible rename or default change and therefore a migration, not a
   refactor; the config field tables already give callers a single spelling, so
   the cost of leaving them is now documentation rather than confusion.  If
   wanted: add the new names as keyword aliases, warn through
   `lumenairy/_deprecation.py` on the old ones for one release, then drop.
   **Effort ~3 h each**, plus a CHANGELOG migration note.

4. **A covering array over `dataclasses.fields()`.**  The audit's win #2 ("a
   covering array over dataclass fields is trivial to write") is now possible
   but not written: WP-A15a's array is over hand-declared factor groups, and
   regenerating it from `lens_config`'s tables would let the legal-combination
   lattice be derived rather than transcribed.  The obstacle is that the
   exclusion lattice is about VALUES, not fields (`surface_model='displaced'`
   is incompatible with `fresnel=True`, not with `fresnel` as such), so the
   fields alone do not generate it.  The real win is a `levels=` annotation per
   field (`dataclasses.field(metadata={'levels': (...)})`) feeding WP-A15a's
   `_pairwise_rows` directly.  **Effort ~1 d**, and it should replace the
   hand-written factor tables rather than sit beside them.

5. **The re-dispatch's `stacklevel` offset.**  See the residual risk in
   section 2.1.  A proper fix threads an offset through ~40 `warnings.warn`
   sites in `_lens_real` / `_lens_traced`; the symptom is a warning attributed
   one frame too shallow, only for callers who used a config object.
   **Effort ~2 h**, low value.

---

## 8. Post-commit follow-ups (three items routed after `c7c9ebbb`)

### 8.1 `mypy --strict lumenairy/__init__.py`: **33 errors -> 0**

WP-A21 held the root `__init__` out of the mypy whitelist at 33 errors,
attributing them to WP-A16's in-flight exports.  Measured breakdown of the 33
and what each needed:

| count | error | cause | fix |
|---|---|---|---|
| 30 | `Module "lumenairy.elements.lenses" does not explicitly export attribute X` | `lenses.py` is a compatibility SHELL: it re-imports 30 names from `_lens_jax` / `_lens_real` / `_lens_thin` / `_lens_traced` / `_lens_traced_multibranch` / `_lens_traced_uniform` / `lenses_gbd` so the pre-v3.5.5 import paths keep working, and `--strict` sets `no_implicit_reexport` | every one respelled `X as X` -- the PEP 484 marker for a deliberate re-export (`lenses.py:1038-1180`) |
| 1 | same, for `apply_real_lens_maslov` | single-name import | same |
| 1 | `Module "lumenairy.raytrace" does not explicitly export attribute "trace_world"` | `raytrace/__init__.py:79` imports `trace_world` but its `__all__` (`:124`) omits it.  **No spelling on the consumer side can fix this** -- the rule is about the SOURCE module | imported from its DEFINING module instead: `from .raytrace.world_trace import trace_world as trace_world` (`world_trace.__all__` does list it).  Identity verified: `la.trace_world is rt.trace_world is world_trace.trace_world` -> True |
| 2 | `Function is missing a type annotation` | the PEP 562 `__getattr__` / `__dir__` pair at `lumenairy/__init__.py:2205,2237` (A15b's lazy loader) | annotated `-> _Any` / `-> _List[str]` with a written reason for the width |

**The two PEP 562 pairs the follow-up named, and a third.**
`lumenairy/backend/__init__.py:56,86` (mine) is now
`__getattr__(name: str) -> ModuleType` -- narrower than `Any` on purpose, and
the docstring says why: `'scipy'` is the only name it resolves and it is always
a module.  `lumenairy/__init__.py` gets `-> _Any`, also with the reason
recorded: its two halves return a dtype/int/str knob value on one branch and
one of 63 solver objects on the other, which share no useful type.  I annotated
`lumenairy/elements/__init__.py:234,258` as well -- same pair, same file
ownership, and it was the only thing keeping that module off the whitelist.

`typing` is imported under **underscore** names (`_Any`, `_List`) in both
package `__init__` files: those namespaces ARE the public API, and
`lumenairy.Any` would be a name every surface walker had to learn to ignore.
Verified: `'Any' in dir(lumenairy)` and `'Any' in dir(lumenairy.elements)` are
both False.

While there I also annotated the five remaining strict gaps in my own
`lens_config.py` (`_require_choice`'s `choices`, `field_names`'s `out`,
`_signature_info`'s and `resolve_entry_point_kwargs`'s `fn`, `_wants_config`'s
four parameters).

**Measured after, all four:**

```
lumenairy/__init__.py                        Success: no issues found in 1 source file
lumenairy/backend/__init__.py                Success: no issues found in 1 source file
lumenairy/elements/lens_config.py            Success: no issues found in 1 source file
lumenairy/elements/__init__.py               Success: no issues found in 1 source file
```

`ruff check lumenairy/` clean; 233 + 75 ids green on the export and lazy-load
walkers (`test_public_api`, `test_v4_16_0_walker_all_symmetry`,
`test_v5_2_walker_shell_vs_canonical`, `test_v5_2_walker_pep562_forwarding`,
`test_audit2609_a15b_reexports`, `test_audit2609_a15b_lazy_and_layering`,
`test_audit2609_a15a_packaging`, the three a16 files, the cache-registry pin).

**One cost, and a one-line way to remove it.**  Ruff's isort (with the project's
default `combine-as-imports = false`) splits every aliased member onto its own
`from X import (...)` statement, so `lenses.py`'s re-export block grew from 8
statements to 31 (+~60 lines).  The comment blocks stayed attached to the first
statement of each group and `ruff check` is clean, but if the orchestrator
prefers the compact form, **`combine-as-imports = true` under
`[tool.ruff.lint.isort]` in `pyproject.toml`** collapses all 31 back into 8 with
no source change.  I did not touch `pyproject.toml`.  (The other route -- an
`__all__` on `lenses.py` -- was rejected: the module has none today, so an
`__all__` listing only the 31 re-exports would silently narrow
`from lumenairy.elements.lenses import *`, which currently exports every public
name in a 1 180-line module.)

### 8.2 `lens_config._VOCAB_CACHE` enrolled with the cache registry

`test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py::test_every_cache_owning_module_enrolls_with_registry`
was red:

```
lumenairy/elements/lens_config.py:147 _VOCAB_CACHE (cache_dict): cache owner
does not call ``register_cache_clearer(...)`` anywhere in the module.
```

Added `clear_lens_config_vocabulary_cache()` and the house registration block
(late-binding lambda, copied from `analysis/beam_stats.py`) at
`lens_config.py:171-208`, registered as **`lens_config_vocabulary`**.

The docstring records why the clearer exists, because it is NOT a memory
measure: the cache holds four short tuples borrowed from `_lens_real` at first
use (tens of bytes), and the reason it must be drainable is that they are
BORROWED -- a test or an `importlib.reload` that swaps one of those vocabularies
would otherwise be validated against the pre-swap copy for the life of the
process.  Refilling is always safe.

Measured: registry **18 -> 19** clearers; `clear_asm_caches()` drains
`_VOCAB_CACHE` (4 entries -> 0) and `_vocab()` refills correctly on the next
call; `test_v4_16_1_...` + `test_niche_r0_byte_budgeted_cache` **35 passed**.

The other half of item (2) is closed as WP-A22 found it: `_same()` has carried
one `try` with `except (TypeError, ValueError)` -- zero broad clauses -- since
the except-budget gate caught the first draft mid-WP-A16 (section 4).  I have
not changed it since.

### 8.3 `test_niche_d6_exact_tilted_leg::test_decentred_carrier_decentre_penalty_envelope` -- NOT a WP-A16 regression

**Measured, not argued.**  The library at each commit was extracted READ-ONLY
with `git archive <rev> lumenairy` into the scratchpad and run against the
CURRENT test file, so only the library varies.  Nothing was checked out,
stashed or written to the repository.

| commit | subject | verdict | on-axis EE2 ratio |
|---|---|---|---|
| `a4e8e855` | `fix(pmm): four cross-build names` (pre-campaign base) | **passed** | > 0.97 |
| `6dfc79d6` | `feat(traced): exact gap kernel on all backends, spline Newton default` | failed *(a different assertion; `r_on` not reached)* | -- |
| `4e8ea247` | `feat(lens): banded ray-density + inverse-characteristic evaluator` | **passed** | > 0.97 |
| `e3f7185a` | `docs(lens): the three corrections from the 5.44.0 follow-ups` | **passed** | > 0.97 |
| **`a18ab074`** | **`fix(carrier): WP-A6 -- traced-carrier chain: focus readout sized from the BEAM (C1), fit radius about the beam centre (C2), complex64 preserved (C3), separable phases and a two-grid transfer function (C4)`** | **failed** | **0.9698** |
| `b97c0b6e` | `fix(lens-analytic): WP-A2` | failed | 0.9698 |
| `37d8afe7` | `fix(lens-traced): WP-A3` | failed | 0.9698 |
| `949edb3b` | `chore(hygiene): WP-A21` -- **the commit immediately before mine** | failed | **0.9698** |
| `2ede9a16` | HEAD | failed | **0.9698** |

**The crossing is `a18ab074` (WP-A6)**, ten commits and two days before
`c7c9ebbb`.  `e3f7185a` -- its immediate parent -- passes; `a18ab074` fails at
0.9698; and the value is then **bit-stable at 0.9698 through every commit
since**, including the one immediately before mine.  Running the same test
against the pre-WP-A16 tree gives `EE2 ratio 0.9698` to the same four decimals
as HEAD.  Two consecutive runs at HEAD also give 0.9698, so this is not the
threaded-QR non-determinism the run warns about.

**WP-A16 therefore did not move this number**, and the follow-up's premise
("the only non-comment changes on its path are yours") does not hold:
`git log a4e8e855..949edb3b -- lumenairy/propagators/carrier.py
lumenairy/elements/_lens_traced.py lumenairy/elements/_lens_real.py` lists
**20+ library commits**, every one of them this campaign's own deliberate
physics corrections (L1-L20, T1-T16, C1-C5 and their VERIFY passes).

**Why WP-A6 is the plausible mechanism, not a defect.**  The failing quantity is
`m_on['ee'][2.0] / o_on['ee'][2.0]` -- the chain's encircled energy inside a
2 um radius over the oracle's -- and WP-A6's headline C1 is *"focus readout
sized from the BEAM"*.  Resizing the readout window is exactly what moves an EE
at a fixed radius.  I did not verify the mechanism (that is WP-A6's subsystem,
not mine); the commit is named so its owner can.

**The bar has no derivation, and it fails by 2.2e-4.**  Full metrics measured at
HEAD:

| quantity | bar | recorded 2026-07-29 | measured now | |
|---|---|---|---|---|
| `r_on` | `> 0.97` | 0.9966 | **0.969787** | **fails by 2.2e-4 (0.02 %)** |
| `r_off` | `0.965 < r < 1.005` | 0.9828 | 0.985518 | passes |
| oracle FWHM off/on | `< 0.10` | -- | 0.0 | passes |
| chain/oracle FWHM off | `< 0.05` | -- | 0.0 | passes |
| chain/oracle FWHM on | (not asserted) | -- | 0.0 | -- |
| EE2 on axis | -- | -- | chain 0.759860, oracle 0.783533 | |

`r_on > 0.97` is a **per-build number with no derivation**: a one-sided
threshold placed 0.0266 below a single 2026-07-29 measurement, with no oracle
error floor, no decade analysis and no statement of what magnitude of defect it
is meant to catch.  Its sibling `r_off` is a proper two-sided envelope with a
written rationale for both directions -- and `r_off` still passes.

**I did not edit the test.**  `tests/unit/test_niche_d6_exact_tilted_leg.py` is
not in WP-A16's ownership list, and restating a physics envelope for a
subsystem WP-A16 did not change -- on a crossing this bisect attributes to
WP-A6 -- is exactly the cross-package edit the ownership rule exists to
prevent.  The restatement is specified for its owner in section 5 item 7,
together with a second finding the investigation turned up.

### 8.4 A finding the d6 investigation turned up: the shipped calibration table is stale

`propagate_traced_carrier_chain` (`carrier.py:8986`) emits a `RuntimeWarning`
to USERS that quotes a measured decentre calibration:

> MEASURED on the K=-n^2 conic stand-in ... (EE2 ratio): 0.00 w -> 0.997;
> 0.25 w -> 1.002; 0.50 w -> 1.005; 0.75 w -> 0.977; 1.00 w -> 0.983;
> 1.50 w -> 0.923.

Measured today on that same stand-in: **0.00 w -> 0.9698**, **1.00 w -> 0.9855**.
Both ends have moved, and they have **crossed over**: the decentred ratio is now
the BETTER of the two, which inverts the premise of both the warning and the
test's own docstring ("the chain tracks the oracle on axis and slightly worse
when the same beam is decentred").  A shipped warning that quotes stale measured
numbers is the class this audit exists to close.  `carrier.py` is not mine --
request in section 5 item 8.

---

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A16_CHANGELOG.md`

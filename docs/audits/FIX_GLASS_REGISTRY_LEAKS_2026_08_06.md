# Fix: GLASS_REGISTRY module-scope leaks (2026-08-06)

Closes the last two failures in the 11,462-test release preflight for branch
`feat/pmm-per-layer-roadmap`:

    tests/unit/test_niche_d8_congruence_workers.py::test_snapshot_is_picklable
    tests/unit/test_niche_d8_congruence_workers.py::test_a_clean_glass_snapshot_reports_nothing_unpicklable

Both passed in isolation and failed in serial full-suite order.  Tests-only
change; `lumenairy/**` is untouched.

## 1. Diagnosis (verified, not assumed)

`lumenairy.glass` keeps its material tables -- `GLASS_REGISTRY`,
`SELLMEIER_COEFFICIENTS`, `GLASS_VALIDITY` -- as process-global dicts.  Test
modules that need a dispersionless MODEL glass (so their oracle is a closed-form
number rather than a Sellmeier fit) wrote it straight into `GLASS_REGISTRY` at
MODULE SCOPE and never removed it:

    GLASS_REGISTRY['_G1CACHE'] = lambda wl: 1.5168     # tests/unit/test_g1_cache_memory.py:24

pytest imports every selected test module during COLLECTION, before the first
test runs, so a single such line poisons the tables for the whole session -- and
a lambda does not pickle.  D8 asserts that a CLEAN worker-state snapshot has
nothing unpicklable in it, because a model glass cannot cross a process boundary
and the parallel congruence path must therefore degrade to serial and say so.
That assertion is correct; the state it was handed was not clean.

Fail-before, deterministic, one command:

    $ python -m pytest tests/unit/test_g1_cache_memory.py \
          tests/unit/test_niche_d8_congruence_workers.py -q -p no:randomly

    E  _pickle.PicklingError: Can't pickle <function <lambda> at 0x...>:
       it's not found as tests.unit.test_g1_cache_memory.<lambda>
       when serializing dict item '_G1CACHE'
       when serializing dict item 'GLASS_REGISTRY'
       when serializing dict item 'glass'
    tests\unit\test_niche_d8_congruence_workers.py:241: PicklingError

    1 failed, 1 passed

The library is NOT at fault.  `_multi_capture_worker_state`,
`_multi_unpicklable_glass` and `_multi_apply_worker_state`
(`lumenairy/propagators/carrier.py:8387-8516`) each behave as specified; the
snapshot faithfully reported the polluted process it was asked to describe.
This is test hygiene, the same class as the `USE_PYFFTW` leak fixed in 5.32.1
and the same class the niche C11 flag guard closes for scalar mode flags.

## 2. Polluter enumeration

Enumerated mechanically, not by grep alone: a scanner imported all 460
`tests/unit/test_*.py` modules one at a time in a single interpreter and diffed
the three glass tables after each import (scratch script, not committed).

**29 modules, 41 keys, 28 of them unpicklable callables.**

| # | module | keys added at import scope | shape |
|---|--------|----------------------------|-------|
| 1 | `test_audit_lens_models_2026_07.py` | `_C1_ABCD_GLASS` | `setdefault`, aliased import |
| 2 | `test_fga.py` | `_GATE_N1p5168` | `setdefault` |
| 3 | `test_g1_cache_memory.py` | `_G1CACHE` | plain assign |
| 4 | `test_g1_gate_generality.py` | `_G1_1p5168`, `_G1_1p62`, `_G1_1p70` | module-scope `for` loop |
| 5 | `test_g2_displaced_congruence.py` | `_G2_A`, `_G2_B` | plain assign |
| 6 | `test_hammer_h1_slant_obliquity.py` | `_H1_FIX_GLASS` | plain assign |
| 7 | `test_hammer_h2_displaced_projection.py` | `_H2_DISP_GLASS` | plain assign |
| 8 | `test_hammer_h3_traced_nyquist_guard.py` | `_H3_FIX_GLASS` | plain assign |
| 9 | `test_hammer_h6_traced_carrier_eikonal.py` | `_H6_FIX_GLASS` | plain assign |
| 10 | `test_hammer_h7_gbd_diverging.py` | `_H7_FIX_GLASS`, `_H7_FIX_162` | two sites ~200 lines apart |
| 11 | `test_niche_k1_kmah_caustic.py` | `_K1A`, `_K1CAU` | plain assign |
| 12 | `test_niche_k3_perf.py` | `_K3A` | plain assign |
| 13 | `test_niche_k4_uniform_caustic.py` | `_K4CAU` | plain assign |
| 14 | `test_niche_p1_gbd_chain.py` | `_N4_FIX_GLASS`, `_N4_FIX_162` | plain assign |
| 15 | `test_niche_p1_traced_tiltaware.py` | `_N5_FIX_GLASS` | plain assign |
| 16 | `test_niche_p10_transverse_walk_remap.py` | `_P10A` | plain assign |
| 17 | `test_niche_p11_ray_density_amplitude.py` | `_P11A`, `_P11SLOW`, `_P11CAU` | plain assign |
| 18 | `test_niche_p2_displaced_extreme.py` | `_P2_A`, `_P2_B` | plain assign |
| 19 | `test_niche_p3_pointwise_obliquity.py` | `_P3A` | plain assign |
| 20 | `test_niche_p4_gbd_reexpand.py` | `_P4_GLASS` | plain assign |
| 21 | `test_niche_p5_sampling.py` | `_P5_GLASS` | plain assign |
| 22 | `test_niche_p7_seidel_gate.py` | `_P7_1p5168`, `_P7_1p62`, `_P7_1p70` | loop + `setdefault` |
| 23 | `test_niche_p8_capstone.py` | `_P8A`, `_P8B` | plain assign |
| 24 | `test_niche_p9_decenter_tilt.py` | `_P9A` | plain assign |
| 25 | `test_niche_r1_cosgrid_cache.py` | `_R1A` | plain assign |
| 26 | `test_niche_r2_pearcey_cusp.py` | `_K4CAU` | plain assign, COLLIDES with #13 |
| 27 | `test_niche_r3_gbd_mem_lstsq.py` | `_R3A` | plain assign |
| 28 | `test_niche_r4_fga_dual_vectorize.py` | `ZF_R4` | plain assign |
| 29 | `test_niche_r5_gbd_vector_catastrophe.py` | `_R5_GLASS`, `_R5_162` | plain assign |

Note #26: `test_niche_r2_pearcey_cusp.py` and `test_niche_k4_uniform_caustic.py`
both registered `_K4CAU` globally.  Whichever imported second silently rebound
the other module's glass object -- the values happened to agree (1.5168), so it
never showed.  Per-module ownership removes the collision by construction.

### Run-time (fixture / test-body) mutations without teardown

A second, smaller class: registrations inside a fixture or test body that never
undo themselves.  These do not break D8 in the observed order but are the same
defect one scope down.  Both were fixed at the source, to the same
`MODULE_GLASSES` idiom, so `LUMEN_TEST_GLASS_LEAK_STRICT=1` is green rather
than merely self-healing.

| module | key | site | fix |
|--------|-----|------|-----|
| `test_fga_h4_h5.py` | `ZF` | module-scoped `presc` fixture, no teardown | `MODULE_GLASSES` |
| `test_carrier_referenced.py` | `_CARRIER_HANDOFF_GLASS` | test body, no `try/finally` | `MODULE_GLASSES` + `_N_HANDOFF_GLASS` |

**31 test modules changed in total** (29 import-scope + these 2).

Audited and found ALREADY CLEAN (each has a `try/finally` or `pop` teardown):
`test_audit_glass.py`, `test_audit_p1_glass_registration.py`,
`test_audit_polarization.py`, `test_audit_s4_3_waveoptics_biconic.py`,
`test_audit_w4_glass_registry_meshgrid.py`,
`test_v4_15_agent_e.py`, `test_v4_16_0_agent_d_validity_ranges.py`,
`test_v4_16_1_agent_c.py`, `test_v4_16_3_agent_d.py`, `test_v5_6_glass_memo.py`,
`test_validation_helpers.py`, and the D8 module itself.
`test_niche_r3_gbd_mem_lstsq.py:451-452` registers `_R3B` inside a SUBPROCESS
source string -- a different process, not a leak here, left as-is.

### A third "polluter" that is NOT one: the library's user-fixed namespace

The strict-mode guard immediately caught a module grep had missed:
`test_audit_p1_glass_registration.py` grows `GLASS_REGISTRY` by ~200
`__spherical_<n>` / `__aspheric_<n>` entries.  Investigated rather than
"fixed" -- and it is **intentional library behaviour, not a defect**:

* `lumenairy/raytrace/trace.py:1502,1531` registers a content-derived
  fixed-index pseudo-glass for every numeric `n_lens` handed to
  `surfaces_from_elements`.  Content-derived means idempotent, so the growth is
  bounded by design (audit P3-61); the module in question exists precisely to
  PIN that (`_build(1.5432)` twice, then `assert len(GLASS_REGISTRY) ==
  before_reg`).
* Their dispatch value is `_USER_FIXED_SENTINEL = ('__user__', '__fixed__',
  '__fixed__')`, and `glass._clear_glass_caches` (`lumenairy/glass.py:1716-1731`)
  deliberately PRESERVES exactly this set, because for these names
  `glass._glass_cache` is the AUTHORITATIVE value store rather than a cache:
  dropping the registry row strands the cached `_FixedIndex` and the next
  `get_glass_index` raises *"flagged as user-fixed but has no _glass_cache
  entry"*.

Verified live: building one spherical element registers
`'__spherical_1.5432' -> ('__user__', '__fixed__', '__fixed__')`.

Both guards therefore carve this namespace out by VALUE (`_is_library_user_fixed`),
not by name pattern -- a guard that reverts the library's own documented policy
would be a worse bug than the one it closes.  These entries are also plain
tuples, so they pickle and were never capable of breaking D8.

## 3. The fix

### 3.1 Ownership, not detection: `MODULE_GLASSES` + one shared fixture

`tests/conftest.py` gains a single module-scoped autouse fixture,
`_module_glass_registry_guard`.  A module that needs a model glass DECLARES it
and does not register it:

    # before -- process-global, forever
    from lumenairy.glass import GLASS_REGISTRY
    GLASS_REGISTRY['_G1CACHE'] = lambda wl: 1.5168

    # after -- exists exactly where it is used, and nowhere else
    MODULE_GLASSES = {'_G1CACHE': lambda wl: 1.5168}

The fixture reads `request.module.MODULE_GLASSES`, `update`s the entries into
`GLASS_REGISTRY` IN PLACE (other modules hold that dict by reference, so it must
never be rebound), and on teardown restores all three tables to the exact
snapshot it took at setup -- which also reverts anything a test body or fixture
in that module mutated and forgot to undo.

MODULE scope matches the existing niche C11 `_module_flag_leak_guard`: the
registration has to outlive a module's own module-scoped chain fixtures (which
build prescriptions naming the model glass), and pytest sets autouse fixtures up
before non-autouse ones at the same scope, so the glass exists before anything
can resolve it.

ONE shared fixture keyed off a module-level name, not 29 copies of a per-module
fixture: a copied fixture is a copied defect surface, and the next module that
needs a model glass should write one dict entry and get the cleanup for free.
Restoring SILENTLY matches the flag guard -- the goal is an order-independent
suite, not a red one.  `LUMEN_TEST_GLASS_LEAK_STRICT=1` fails the leaking module
instead, which is how you find a culprit once you know the class is live.

Per-module diff shape (all 29): the module-scope write became a `MODULE_GLASSES`
dict at the same line, and the now-unused `from lumenairy.glass import
GLASS_REGISTRY` was dropped in the 24 modules that had no other use of it.  The
five that still use the registry at run time keep the import:
`test_g2_displaced_congruence.py`, `test_niche_p2_displaced_extreme.py` and
`test_niche_p7_seidel_gate.py` (`_n()` index helpers), plus the two loop-shaped
declarations which became dict comprehensions.

### 3.2 Permanent leak detector

Two hooks, because the two leak classes need different mechanisms:

* **Run-time leaks** -- `_module_glass_registry_guard` (above) restores the
  tables at module teardown and, under `LUMEN_TEST_GLASS_LEAK_STRICT=1`, raises
  naming the table, the key, and the `__module__` of the callable that added it.

* **Import-time leaks** -- a module-scoped fixture structurally CANNOT catch
  these: pytest imports every selected module during collection, so the
  pollution is already present when the first module guard takes its snapshot,
  and the guard would preserve it as part of the baseline.  So
  `pytest_collection_modifyitems` in `tests/conftest.py` diffs the tables
  against `_PRISTINE_GLASS` -- captured at conftest import, the only baseline
  that predates collection -- and REMOVES what a test module added.

  Removing rather than only reporting is the point: it makes the next module to
  register at import scope fail ITS OWN tests (its model glass is gone by the
  time they run) instead of silently failing an unrelated module's picklability
  assertion three thousand tests later.  Entries that are picklable AND not
  attributable to a test module are reported but left in place -- that shape
  would be the library lazily loading a catalogue, which is not this hook's
  business.

Both guards skip `_USER_FIXED_SENTINEL` entries (`_is_library_user_fixed`) for
the reason in section 2 -- that namespace is library-owned and
`glass._clear_glass_caches` preserves it on purpose.

Cost: three shallow dict copies (78 + 52 + 77 entries) per module setup and
teardown, and one `pickle.dumps` per import-time offender (zero on a clean run).
No imports, no I/O.  Not measurable against a 20-minute suite.

Detector verified against a deliberately-planted probe module (created,
exercised, deleted):

    # A) default (permissive): polluter fails ITS OWN test, D8 stays green
    IMPORT-TIME glass-table pollution detected after collection:
      GLASS_REGISTRY['_ZZ_IMPORT_TIME_LEAK'] from tests.unit.test_zz_tmp_leak_probe
      -- UNPICKLABLE. Declare model glasses in a module-level MODULE_GLASSES dict
      ... REMOVED: GLASS_REGISTRY['_ZZ_IMPORT_TIME_LEAK'].
    FAILED tests/unit/test_zz_tmp_leak_probe.py::test_probe_import_time_glass_is_gone_by_run_time
    1 failed, 31 passed             <-- all 30 D8 tests green

    # B) LUMEN_TEST_GLASS_LEAK_STRICT=1, import-time leak
    ERROR: IMPORT-TIME glass-table pollution detected after collection: ...

    # C) LUMEN_TEST_GLASS_LEAK_STRICT=1, test-body leak
    ERROR at teardown of test_probe_leaks_from_a_test_body
    AssertionError: this module leaked lumenairy.glass table entries to every
      LATER test in the process (glass-registry leak guard); ...
      GLASS_REGISTRY['_ZZ_TEST_BODY_LEAK'] ADDED by tests.unit.test_zz_tmp_leak_probe

## 4. Evidence

### 4.1 Import-scan, before and after

    # before
    460 modules to import
    MODULE-SCOPE POLLUTERS: 29
    modules leaving UNPICKLABLE entries: 28
    total keys added across all imports: 41

    # after
    460 modules to import
    MODULE-SCOPE POLLUTERS: 0
    modules leaving UNPICKLABLE entries: 0
    total keys added across all imports: 0

### 4.2 Reproducer, pass-after

    $ python -m pytest tests/unit/test_g1_cache_memory.py \
          tests/unit/test_niche_d8_congruence_workers.py -q -p no:randomly
    34 passed, 56 warnings in 20.54s

### 4.3 Mini-sweep -- all 29 import-scope modules + D8, one serial invocation

    $ python -m pytest -q -p no:randomly -x <29 modules> \
          tests/unit/test_niche_d8_congruence_workers.py
    393 passed, 129 warnings in 2267.09s (0:37:47)

### 4.4 Glass-consumer sweep -- STRICT mode, one serial invocation

14 modules that mutate or read the glass tables (grep for `GLASS_REGISTRY`
consumers), plus `test_g1_cache_memory.py` and D8, with
`LUMEN_TEST_GLASS_LEAK_STRICT=1` so any surviving leak ERRORS:

    # before the two run-time source fixes + the user-fixed carve-out
    290 passed, 2 skipped, 3 errors in 265.46s
      ERROR ... test_fga_h4_h5             GLASS_REGISTRY['ZF']
      ERROR ... test_carrier_referenced    GLASS_REGISTRY['_CARRIER_HANDOFF_GLASS']
      ERROR ... test_audit_p1_glass_registration  ~200 __spherical_* (library-owned)

    # after
    294 passed, 2 skipped, 62 warnings in 463.27s (0:07:43)      0 errors

### 4.5 Full-suite import order -- the actual preflight condition

The cheapest faithful proxy for the release preflight: collect the WHOLE unit
suite (so all 460+ modules are imported, in preflight order) and run only the
two tests that were failing, under STRICT mode.

    $ LUMEN_TEST_GLASS_LEAK_STRICT=1 python -m pytest tests/unit -q -p no:randomly \
        -k "test_snapshot_is_picklable or test_a_clean_glass_snapshot_reports_nothing_unpicklable"
    2 passed, 11538 deselected in 19.00s

    $ LUMEN_TEST_GLASS_LEAK_STRICT=1 python -m pytest tests/unit -q -p no:randomly --collect-only
    11538/11540 tests collected (2 deselected) in 54.87s

No `IMPORT-TIME glass-table pollution` line and no strict-mode `UsageError`
across the full 11,540-test collection -- i.e. all 460+ modules import without
touching the material tables.

### 4.6 Lint

    $ python -m ruff check tests/
    All checks passed!

Mid-task caveat, recorded because it is someone else's change and not this
one's: an earlier run of the same command reported 3 `I001` import-sort errors
in `test_niche_exact_gap_kernel.py` (x2) and `test_niche_p2_design_battery.py`.
Neither file is touched by this change; both were rewritten at 19:34 local by
something outside this task (editor auto-fix or a concurrent session), and the
errors are gone.  Flagged rather than claimed.

`F401` is already per-file-ignored for `tests/**` in `pyproject.toml`, so
dropping the now-unused `GLASS_REGISTRY` imports was cosmetic, not required.

## 5. Not closed

* **No full 11,462-test preflight was re-run here.**  The evidence is the
  full-suite-collection run in 4.5 (which reproduces the exact import order the
  failure needed), the 393-test mini-sweep, and the 294-test strict sweep.  The
  preflight itself should still be the gate.
* **Only `tests/unit` was scanned** for import-time pollution.
  `tests/integration/` runs `validation/` as subprocesses under its own
  `validation/conftest.py`, which this change does not touch; `testpaths` is
  `["tests"]`, so `tests/conftest.py`'s hook does cover
  `tests/integration/` if it ever grows a module-scope registration.
* **`test_niche_exact_gap_kernel.py` and `test_niche_p2_design_battery.py`
  changed on disk at 19:34 local**, mid-task, by something that is not this
  change (see 4.6).  Their `I001` errors cleared as a result.  Worth confirming
  nobody else was mid-edit in the same working tree.
* **The guards cover the three glass tables only.**  Other process-global
  containers (`_glass_cache`, the FFT plan caches, `_TRACED_KWARG_DEFAULTS_CACHE`)
  are covered by the niche C11 guard or by `shipped_fft_dispatch`, not by this
  one.  A leaked `_glass_cache` entry whose registry row this guard removes is
  harmless (the value is recomputed on the next registration) but is not
  actively restored.
* **`MODULE_GLASSES` is a convention, not a schema.**  Nothing forces a new
  module to use it -- but the collection hook makes the alternative
  (module-scope registration) fail that module's own tests immediately, which
  is the enforcement mechanism.

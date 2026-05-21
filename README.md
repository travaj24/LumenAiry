# lumenairy

[![PyPI](https://img.shields.io/pypi/v/lumenairy.svg)](https://pypi.org/project/lumenairy/)
[![Validate](https://github.com/travaj24/LumenAiry/actions/workflows/validate.yml/badge.svg)](https://github.com/travaj24/LumenAiry/actions/workflows/validate.yml)
[![Python](https://img.shields.io/pypi/pyversions/lumenairy.svg)](https://pypi.org/project/lumenairy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python library for coherent optical beam propagation and
manipulation using the Angular Spectrum Method (ASM) and related techniques.

**Author:** Andrew Traverso

## What's new in 5.1.0

**Major structural release.**  v5.1 lands the two long-deferred items
from the v5.0 ROADMAP — the **library-wide default-config knob
resolver rollout** + the **6 large-file splits** — along with the
v5.0.1 audit closure.  7 agents in parallel disjoint scopes (A:
resolver rollout; B-G: 6 file splits) + a Wave-4 integration sweep
that closed cross-agent test breakage.

**Zero physics regressions in 8 consecutive releases.**

**3628 unit pass / 5 skip / 1 xfail = 3634 collected** (was 2895 at
v5.0.1; +733 net); **34/34 validation pass**.

### Default-config knob resolver rollout

`set_default_wave_propagator(...)`, `set_default_dy(...)`, and
`set_default_real_dtype(...)` — shipped as API-only stubs in v4.16.2
— now actually steer downstream behaviour at `apply_real_lens`,
`apply_real_lens_traced`, `propagate_through_system`, and
`propagate_ensemble`.  The v4.16.3 no-consumer UserWarning latches
are **retired** in v5.1.  v4.16.3's sibling-gap pin is now an
inverse pin asserting the resolvers ARE consumed.

### 6 large-file splits (no public API change)

Six monolithic >2200 LOC files split into ~35 topical submodules via
re-export shells:

| Original (pre LOC) | New submodules |
|---|---|
| `raytrace/core.py` (4443) | surface / intersection / trace / world_trace / seidel / ray_fan / layout |
| `propagators/propagation.py` (4103) | fft_infra / asm / fresnel / rs / sas / mft |
| `propagators/asymptotic.py` (4561) | asymptotic_{modes / canonical_fit / aberration_tensor / maslov / jax_twin} |
| `optimize/core.py` (4538) | parameterizations / merit_terms / wrapper_merits / context / driver / jax_merits |
| `io/prescriptions.py` (3224) | prescriptions_{builders / zemax / code_v / quadoa / transforms} |
| `analysis/core.py` (4088) | beam_stats / strehl / psf_mtf_otf / polychromatic / zernike / opd |

Each split ships per-submodule regression tests verifying public-API
survival via both old and new import paths.

### v5.0.1 audit closures (11 items)

* **`publish.yml` release-process gate** (P1) — `verify` job runs
  the unit suite on the tag's source before `build` + `publish`.
  v5.0.0/v5.0.1 ship-before-CI-green pattern structurally retired.
* Python 3.10 re-added to unit-tests CI matrix.
* 3 shim-removal pins gained Migration-Guide content-lock assertions.
* `test_examples_output_dir` source-inspection tightened.
* Stale `@_skip_no_qt` doublings + comments cleaned up.

### Test counts

A=17, B=141, C=122, D=85, E=14, F=87, G=217. Sum: **683 new tests
across the 7 agents**.  Plus Wave-4 integration fix-ups + v5.0.1
audit pins.  Final **3628 pass / 5 skip / 1 xfail = 3634 collected**;
34/34 validation.

## What's new in 5.0.1

**Closes the v5.0.0 audit through P3.**  Zero P0; 3 P1 + 5 P2 + 8 P3
across infrastructure (lint baseline, benchmarks drift, stale "v5.0"
warning text, missing anti-regression pins, stale docstrings,
ROADMAP drift).  **Zero physics regressions in 7 consecutive
releases.**  3 agents, disjoint scopes.  **2889 unit pass / 5 skip /
1 xfail = 2895 collected**; **34/34 validation pass**.

### P1 closures (3)

* **`set_default_*` UserWarning text v5.0 → v5.1.**  The v4.16.3
  warning bodies said "Consumer wiring staged for v5.0" but the
  v5.0 release deferred to v5.1.  At v5.0 HEAD users saw a warning
  promising the bug was fixed "in v5.0" — *which IS v5.0*.
  v5.0.1 updates the warning text + pinning test to cite v5.1.
* **`benchmarks/test_bench_jax_jit.py` double-break fixed** —
  used both the old `lumenairy.system` path (now
  `lumenairy.propagators.system`) and the pre-v4.12 JAX aperture
  schema (now removed).  Benchmarks aren't in unit-CI scope so
  this slipped through v5.0.
* **CI lint baseline: 4 real F821 forward-reference bugs fixed**
  in `lumenairy/algebra/base.py` and `lumenairy/propagators/system.py`
  via proper `TYPE_CHECKING` guards.  Lint job promoted to
  advisory mode (`continue-on-error: true`); cosmetic 692-error
  cleanup is a v5.1 mechanical-work item.

### P2 closures (5)

* `simulate_detector_image` → `apply_detector` doc-naming
  consistency (the function is `apply_detector`; old name never
  existed).
* **5 shim-removal anti-regression pins** added — v5.0 only
  pinned `lumenairy.analysis.analysis`; the other 4 closures
  (`lumenairy.ao`, `lumenairy.io.hdf5`, top-level
  `lumenairy.system`, JAX aperture legacy schema,
  `cosmic_ray_rate` kwarg) now have parallel pins.
* ROADMAP v5.1 section refresh — stripped v5.0-shipped items;
  documented the 3 `apply_*_lens` re-exports preserved by design.
* 2 stale docstrings fixed (`detector.py` `cosmic_ray_rate`
  block; `propagators/system.py:582` `la.system.evaluate`
  example).
* `[tool.mypy]` gains `follow_imports = "silent"` (config prep;
  CI activation deferred to v5.1).

### P3 closures (8)

* CHANGELOG `__all__` arithmetic (533 entries via 536 smoke tests,
  not 536 entries).
* Negative counter-pin for `test_public_api.py`.
* Unreachable post-raise code removed.
* MCF deferral clarification (`PartialCoherenceMCF` already
  shipped v4.15.1; v5.1 adds the shorter `lumenairy.MCF`
  top-level alias).
* Python 3.14 classifier dropped (CI is 3.10-3.13).
* Various stale-comment refreshes.

### Test counts

A=4, B=6, C=21.  Sum: **31**.  Final 2889 unit pass / 5 skip / 1
xfail = 2895 collected; 34/34 validation.

## What's new in 5.0.0

**Major release.**  v5.0 is the coordinated breaking-change release:
removes back-compat shims that had been carried 3-9 releases past
their deprecation, bumps the Python floor to 3.10, moves
`lumenairy/system.py` under `propagators/`, and adds the CI
infrastructure (ruff, mypy strict incremental, unit-test PR gate,
public-API smoke test) that the structural cleanup needs.

**Scope discipline:** the v4.16.x ROADMAP scoped a wider v5.0 (file
splits, library-wide resolver rollout, MCF object, formula-3
coefficient ingestion, off-axis conic, 5 new examples, test
consolidation).  Those non-breaking items move to **v5.1+** so the
v5.0 diff stays reviewable.  See `Migration-Guide.md` for v5.0
migration recipes and `ROADMAP.md` for the v5.1 horizon.

### Breaking changes

* **Python 3.10+ required.**  Python 3.9 reached EOL 2025-10.
* **`lumenairy.system` -> `lumenairy.propagators.system`.**  Public
  namespace (`la.propagate_through_system(...)`) unchanged; direct
  imports of the private path require migration.
* **5 back-compat shims removed**: `lumenairy.analysis.analysis`,
  `lumenairy.ao`, `lumenairy.io.hdf5`, the pre-v4.12 JAX aperture
  schema (`radius` / `half_width_x` / `inner_radius` in
  `propagate_through_system_jax`), and the v4.9 `cosmic_ray_rate`
  kwarg in `apply_detector`.  Each now raises
  `ModuleNotFoundError` / `ValueError` / `TypeError` with the
  migration recipe inline.

### CI gates (NEW)

* `pytest tests/unit -m "not integration"` job (fast PR feedback)
  across Python 3.10-3.13.
* `ruff check` lint gate.
* `mypy --strict` incremental adoption.
* `tests/test_public_api.py` smoke test: every `__all__` entry is
  `getattr(lumenairy, name)`-resolvable.

### Test counts

**2858 unit pass / 5 skip / 1 xfail = 2864 collected**; **34/34
validation pass**.

## What's new in 4.16.3

**Closes the v4.16.2 audit (`docs/audits/AUDIT_V4_16_2_2026_05_20.md`)
through P3.**  Audit found zero P0; 2 P1 + 6 P2 + 8 P3 concentrated
around v4.16.2's pre-v5.0 prep features being mostly scaffolding
without real consumers, plus 2 structural-bypass issues inside the
new V11 doc-consistency walker itself.  4 agents in disjoint scopes.
**2327 unit pass / 5 skip / 1 xfail = 2333 collected** (up from 2270
at v4.16.2; +57 net); 34/34 validation.

### P1 closures (2)

* **V11 walker pyproject parsing → `tomllib`.**  v4.16.2's regex
  silently mis-parsed `jax-gpu = ["jax[cuda12]>=0.4.20"]` and
  comment-bracket content in `all = [...]`.  Replaced with
  `tomllib` (Python 3.11+) / `tomli` (3.9/3.10 backport).  Two
  anti-regression pins assert the broken regex + hardcoded version
  tuple don't reappear.
* **Migration-Guide §4.16.2 rewrite.**  The v4.16.2 recipe demo'd
  `set_default_wave_propagator('fresnel')` followed by
  `apply_real_lens(...)` -- but the latter hardcodes `'asm'` and
  doesn't consult the knob.  Replaced with a recipe using
  `set_default_complex_dtype` + `set_default_real_dtype` (knobs
  with real consumers); explicit "API-only in v4.16.2/v4.16.3"
  limitation note added.

### P2 closures (6)

* **`get_default_real_dtype` consumer fixed** -- the v4.16.2
  "representative wiring" was structurally unreachable dead code;
  refactored so the knob is reachable via the
  `getattr(ensemble, 'dtype', None)` default path.
* **`set_default_wave_propagator` + `set_default_dy` one-shot
  `UserWarning`** -- both knobs store values no library code reads
  at v4.16.3; setters now warn that the knob is "API-only;
  consumer wiring lands in v5.0".  Sibling-gap pin asserts zero
  consumers -- FAILS LOUDLY when v5.0 wires the first.
* **V11 version list → CHANGELOG-driven scan.**  Was hardcoded
  `('4.13.0', '4.15.1', '4.16.1', '4.16.2')`; now extracts
  `## [X.Y.Z]` headings from CHANGELOG with breaking-change marker
  detection.  Future v4.17.0+ breaking entries auto-included.
* **V11 → CHANGELOG↔Migration-Guide drift coverage.**  New test
  enforces `>= 200 chars of non-whitespace body` per Migration-Guide
  version section.
* **`Constraint` auto-probe DeprecationWarning** (one-cycle, latched)
  -- pattern-parallel to v4.16.2's MultiWavelengthMerit FutureWarning
  treatment.  Removed in v5.0.
* **`pickle.dumps` probe catch widened** from
  `(PicklingError, AttributeError, TypeError)` to `Exception` --
  catches `RecursionError`, `RuntimeError` from custom `__reduce__`,
  `MemoryError`, and arbitrary `__reduce__` / `__getstate__`
  exceptions.

### P3 closures (8)

* **`__polynomial__` sentinel** parallel to `__sellmeier__` in
  glass.py; polynomial-formula glasses can now opt in to the bundled
  evaluator even with refractiveindex installed.
* **`DEFAULT_REAL_DTYPE` / `DEFAULT_WAVE_PROPAGATOR` / `DEFAULT_DY`
  re-exported at top level** for parity with `DEFAULT_COMPLEX_DTYPE`.
* **Per-surface thickness in high-NA hoist message** (was citing
  `max(thickness)`).
* **Multiprocess / fork-safety documented** for one-shot latches +
  `DEFAULT_*` globals (spawn-mode workers re-import + reset).
* **`psutil>=5.0`** promoted to Required in requirements.txt
  (parity with pyproject.toml hard dep).
* **"Structurally retired" claim softened** in CHANGELOG + ROADMAP
  (honest framing: "all currently-known classes; new classes will
  continue to surface").
* **CHANGELOG sentinel line citation** refreshed `:3015` → `:3032`.
* **POLYNOMIAL/SELLMEIER dispatch order** doc/code reconciled.

### Test counts

A=17, B=16, C=8, D=15, walker-extra=1.  Sum: **57**.  Final
2327 pass / 5 skip / 1 xfail = 2333 collected; 34/34 validation.

## What's new in 4.16.2

**Closes the v4.16.1 audit
(`docs/audits/AUDIT_V4_16_1_2026_05_19.md`) through P3** plus lands
the user-requested **pre-v5.0 prep features**.  Audit found zero
P0; 5 P1 + 8 P2 + 9 P3 split between 3 code-correctness items the
v4.16.1 verifier audit missed (test pins bypassed the production
path) and 4 documentation-surface drifts -- the sibling-gap
meta-pattern v4.16.1 thought it had retired at code surfaces had
migrated to doc surfaces.  4 agents, disjoint scopes.  **2270 unit
pass / 5 skip / 1 xfail = 2276 collected** (up from 2198 at v4.16.1;
+78 net); 34/34 validation.

### Code-correctness P1s (3)

* **`_transfer_jax` high-NA `RuntimeWarning` now reaches users.**
  v4.16.1's `isinstance(np.ndarray)` gate was structurally
  unreachable in production (JAX 0.4+ `jax.Array` isn't a
  `np.ndarray` subclass).  Replaced with a duck-typed gate +
  hoisted eager-only probe at `trace_jax` entry (before the inner
  `jax.jit` wrapper).  End-to-end integration test pins the
  production-path firing.
* **`propagate_ensemble` no longer silently downcasts
  CuPy / JAX -> NumPy.**  `np.asarray(ensemble)` replaced with
  `array_namespace` dispatch via `lumenairy.backend`.  GPU / JAX
  ensembles stay on the backend.
* **`MultiWavelengthMerit` SUM->AVG one-cycle `FutureWarning`.**
  v4.16.1's silent semantics change now alerts users (once per
  process via module-level latch) that existing 3-wavelength
  weight calibrations may need re-scaling.

### Doc-surface P1s + 11th meta-pin walker (4)

* **README `Required` block** rewritten -- `refractiveindex`
  moved to `Optional (via [glass] extras)`; quick-install command
  uses `pip install lumenairy[glass]` instead of force-installing
  the dep.  Test counts headline corrected.
* **requirements.txt** rewritten -- `refractiveindex` dropped;
  `h5py` moved to commented optional section; `zarr>=2.14` -> `>=3.0`;
  full extras-group enumeration in commented form.
* **ROADMAP V9 -> V11** -- V10 (cache-registry walker, shipped
  v4.16.1 but missing from list) + V11 (NEW doc-consistency
  walker, ships this release) added.
* **NEW 11th meta-pin walker** at
  `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py`.
  Scans README / requirements.txt / ROADMAP / CHANGELOG for drift
  against pyproject.toml.  Closes the doc-surface sibling-gap
  meta-pattern structurally.

### Pre-v5.0 prep features

* **Bundled Sellmeier formula-3 (polynomial) evaluator** at
  `lumenairy/glass.py`.  Math infrastructure + dispatch wiring
  ship in v4.16.2 (`_polynomial_index`, `POLYNOMIAL_COEFFICIENTS`
  registry); per-glass coefficient ingestion for the 26 Hikari /
  Sumita / formula-3 CDGM entries is staged for v5.0 (requires
  vendor-source review + n_d cross-check per entry).
* **3 default-config knobs** (parallel to existing
  `set_default_complex_dtype`):
  * `set_default_real_dtype(np.float32 | np.float64)`
  * `set_default_wave_propagator('asm' | 'fresnel' | 'sas' |
    'rayleigh_sommerfeld' | 'rs')`
  * `set_default_dy(None | positive float)`
  * Plus paired `get_default_*` accessors; all 6 functions
    exported at top level.
* **NEW `Migration-Guide.md`** at repo root.  Version-spanning
  migration recipes for v4.13.0 / v4.15.1 / v4.16.1 / v4.16.2 +
  forward-section for v5.0.

### Hygiene + remaining P2/P3

* `Constraint.__post_init__` probe -> opt-in `Constraint.validate()`
  method (was running real user code on every instantiation).
* Lambda detection -> `pickle.dumps` probe (now catches closures +
  `functools.partial`).
* `Constraint` tests in `test_v4_16_0_agent_c.py` migrated to
  module-level functions.
* 10th meta-pin walker hardened to require module-level
  `register_cache_clearer` call.
* `_resolve_bound` 3-tuple guard.
* LM bounds method override (`'lm'` -> `'trf'`) now emits a
  UserWarning.
* `GLASS_VALIDITY` consistency check accepts numpy scalars.
* CHANGELOG sentinel line citation `:2974` -> `:3015` (Agent B's
  ~41-line addition to `optimize/core.py` drifted the
  `_ZERO_APERTURE_MASK` branch).
* CHANGELOG headline arithmetic + UserWarning -> RuntimeWarning
  typo.

### Test counts

A=18, B=16, C=24, D=13+7=20. Sum: 78. Net +78 new tests; final
2270 pass / 5 skip / 1 xfail = 2276 collected; 34/34 validation.

## What's new in 4.16.1

**Closes the v4.16.0 deep audit
(`docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md`) through P3** — the
first audit to actively hunt silent-wrong-answer correctness bugs
alongside the usual structural / UX cleanup.  4 agents, disjoint
scopes.  2198 unit tests pass (up from 2113; +85 net); 34/34
validation pass.

### Correctness bugs (4 fixed)

* **`MultiWavelengthMerit` SUM → AVG** — the wrapper summed instead
  of averaging sub-merit results across wavelengths; documented
  semantics + both sibling classes already divided by `len(...)`.
  Bug was localised to this one class.
* **`shack_hartmann` pitch quantisation** — the slope-to-wavefront
  cumsum used the user-requested `lenslet_pitch` instead of the
  on-grid `sa_pixels * dx`.  At `lenslet_pitch / dx = 1.7` the
  reconstructed amplitude was biased ~17.6% low.
* **`_detect_backend` directory misclassification** — any directory
  was silently routed to Zarr regardless of whether a Zarr store was
  actually present; `pathlib.Path` callers hit `AttributeError`.
  Fix gates on the canonical `zarr.json` (v3) / `.zarray` (v2) marker
  files + `str(path)` cast.
* **LM `bounds` parser** — `(None, 1.0)` is truthy, so `None` leaked
  into the float array and scipy raised an opaque downstream error.
  Explicit `_resolve_bound` helper now routes `None` to `±np.inf`.

### Half-shipped clusters closed

* **`propagate_ensemble(...)` helper** — new
  `lumenairy.propagate_ensemble` iterates a Schell-family
  `(n_realizations, Ny, Nx)` ensemble through any coherent propagator
  and returns `I_partial = <|E_k|^2>_k`.  Memory-efficient
  accumulator default; opt-in `return_ensemble=True` for the full
  stack.  Smoothing factor measured at ~6.95× on the new
  `examples/06_schell_propagation.py`.
* **Default Schell-factory `DeprecationWarning` retired** — 4
  releases of exposure since the v4.15.0 → v4.15.1 return-shape
  change; the warning now produces more confusion noise than
  migration value.  Sentinel + helper preserved as deprecated
  symbols (the v4.15.3 promotion meta-pin imports them); targeted
  for removal in v5.0.
* **JAX-traceable dtype probe** — duck-typed `getattr(E_in, 'dtype',
  None)` so `propagate_through_system_jax` survives `jax.jit` /
  `jax.grad` tracing.
* **High-NA `_transfer_jax` UserWarning** — eager-only guard fires
  at `min |N| < 0.95` (where the paraxial autodiff-stable
  approximation begins to diverge from the NumPy reference); tracer
  path unchanged.

### API + meta-pins

* **`Constraint` API narrowed to scalar-return** + lambda
  `UserWarning` for parallel-workers configs (lambdas crash
  pymoo's `float(_f(xv))` coercion + aren't picklable).
* **10th cache-registry meta-pin walker** — AST-walks every
  `@lru_cache`-decorated module-level function and asserts a paired
  `_cache_registry` enrollment.  15 caches, 0 orphans.  Continues
  the V1–V9 structural-counter-measure trajectory.
* **`_check_glass_registry_consistency()` extends to GLASS_VALIDITY**
  — every validity key must appear in `GLASS_REGISTRY` (and be a
  valid `(λ_min, λ_max)` 2-tuple).

### Compat + glass + UX

* **`refractiveindex` moved from hard deps to `[glass]` extras** —
  aligns the wheel with the lazy-import + `SELLMEIER_COEFFICIENTS`
  fallback already in `lumenairy/glass.py`.
* **`zarr>=2.14` floor bumped to `zarr>=3.0`** — `io/storage.py`
  uses `Group.create_array` (Zarr v3 API); the v2 floor was a
  latent `AttributeError`.
* **4 stale `n_d` inline comments fixed** in `lumenairy/glass.py`
  (H-ZK9B, H-ZF12, D-LAK52, H-ZLAF52A).  No runtime change — the
  actual Sellmeier coefficients were always correct; only the
  comments were stale.
* **`ProcessPoolExecutor` now passes `mp_context='spawn'`** in
  `_lens_traced.py` — matches the README + v4.16.0 claim that
  `spawn` is used (was `fork` on Linux).
* **`examples/06_schell_propagation.py` +
  `examples/07_zemax_load_trace.py`** added.  `CONVENTIONS.md`
  added at the repo root documenting the `create_*` vs `make_*`
  factory verb contract + 9 related conventions.

### Test counts

A=11, B=26, C=20+6, D=22 = 85.  Net +85 new tests; final 2198
unit pass + 5 skip + 1 xfail; 34/34 validation.

## What's new in 4.16.0

**Major minor release** rolling up the entire v4.16 + v4.17 + v4.18
ROADMAP into one ship.  Four large feature buckets: the remaining 4
V4 meta-pin candidates; multi-process atomic-append for `storage.py`;
the full optimisation framework expansion (constrained opt,
checkpoint/resume, Newton-step, multi-objective pymoo); glass +
materials + central cache registry.  2106 unit tests pass (up from
1922; +184 net); 34/34 validation pass.

### Bucket 1 — Structural counter-measures complete (4 V4 meta-pin walkers)

The audit's standing V4 recommendation lands in full.  The library
now ships **9 active dispatcher meta-pins** (cache-clears, cache↔lock,
0+0j, validate_grid_params, check_2d_scalar_field, sentinel-propagation,
xp-dispatch, dy-threading, __all__-symmetry).  The "fix N, miss N+1"
sibling-gap meta-pattern at the public-API surface is now
structurally retired across all known classes.

* **Sentinel-aware branch propagation** — 3 sites discovered, all
  clean.
* **`_xp_of` cross-backend dispatch** — 94 candidates; 5 inline
  fixes in `elements/elements.py`; 56 documented exemptions.
* **`dy` parameter threading** — 36 `apply_*` candidates; 26
  already threading, 10 documented exemptions.
* **`__all__` symmetry** — 752 submodule entries; 717 re-exported;
  35 documented exemptions; 9 inline fixes (8 backend helpers +
  `PYMOO_AVAILABLE` lifted to top-level).

### Bucket 2 — Multi-process atomic-append for `storage.py`

HDF5 SWMR + filelock-based distributed Zarr lock close the
multi-writer story v4.14.3 documented as a single-process
restriction.  `append_plane_h5(..., swmr=True, lock_timeout=30.0)`
+ `_zarr_append_plane(..., lock_timeout=30.0)`.  Subprocess
multi-writer tests via `multiprocessing.get_context('spawn')`
(Linux + Windows portable).  Single-process v4.14.3 atomicity
guarantees preserved bit-for-bit.  Measured overhead: ~5×
slowdown 4-writer contended; <5% lock overhead on large planes
where I/O dominates.  `filelock>=3.0` added to `hdf5` and `zarr`
optional-dependency groups.

### Bucket 3 — Optimisation framework expansion

Four additions to `lumenairy.optimize`:

* **Constrained optimisation** — `Constraint` dataclass +
  `design_optimize(..., constraints=[...])`.  Method-compat
  validator rejects L-BFGS-B / `lm` / `differential_evolution` /
  `basin_hopping` / `dual_annealing` (which silently ignore
  constraints in scipy's core API) and points users at SLSQP /
  trust-constr.
* **Checkpoint / resume** — `design_optimize(..., state_file=...,
  state_save_every=1)`.  JSON state with atomic-replace write;
  resume from persisted `x_best` on restart.  Gated on
  `state_file` non-None for backward compatibility.
* **Multi-objective Pareto via pymoo NSGA-II** —
  `lumenairy.optimize.multi_objective.design_optimize_multi_objective(...)`
  with `ParetoResult` dataclass.  Optional `multi_objective`
  extras group: `pip install lumenairy[multi_objective]`.  Module
  imports unconditionally; only the function call raises
  ImportError if pymoo is missing.
* **Hessian / Newton-step** — `design_optimize(..., method='newton')`
  dispatches to scipy `trust-ncg` with FD-Jacobian-of-FD-gradient
  Hessian.  `UserWarning` recommends L-BFGS-B for `n_params > 30`.

4 new top-level exports: `Constraint`, `ParetoResult`,
`design_optimize_multi_objective`, `PYMOO_AVAILABLE`.

### Bucket 4 — Glass + materials + central cache registry

* **CDGM + Hikari + Sumita Sellmeier catalogues** — 32 new
  glasses; `GLASS_REGISTRY` 46 → 78.  Every entry n_d cross-
  checked against the official datasheet (or refractiveindex.info
  as proxy) at 5e-5 tolerance (matching v4.14.2 S-LAH64/79
  precedent).  Zero cross-check failures.
* **Per-glass Sellmeier validity ranges** — new `GLASS_VALIDITY`
  table with 77 entries.  Extrapolating outside the range emits
  `UserWarning` (does not raise — extrapolation is sometimes
  acceptable for design-space exploration).
* **Central cache registry** — new `lumenairy/_cache_registry.py`
  exposes `register_cache_clearer(name, clear_fn)` /
  `list_registered_cache_clearers()` /
  `clear_all_registered_caches()`.  Retires the lazy-import
  fan-out in `clear_asm_caches`.  9 caches migrated to the
  registry.  `clear_asm_caches` external contract preserved
  bit-for-bit; internal dispatch is now registry-walking instead
  of hand-enumerated.  Structural counter-measure to the cache-
  clear "fix N, miss N+1" pattern.

### Test counts

A=30, B=15, C=13, D=127.  Net +184.  Final: 2106 unit pass + 5
skip + 1 xfail; 34/34 validation.

### ROADMAP status post-v4.16.0

* **v4.16, v4.17, v4.18 — all items shipped**.
* **v5.0 — immediate horizon**.  Major structural release:
  6 file splits, CI gates, remove 8 active back-compat shims,
  shared Chebyshev helpers extraction, audit-fix test-file
  consolidation, off-axis conic in surface frame, bump
  `requires-python` to >=3.10, docs.

### Known issues / flagged for v4.16.1

Bundled Sellmeier formula-3 (polynomial) evaluator — Hikari,
Sumita, and 4 CDGM glasses use refractiveindex.info formula 3.
v4.16.0's `_sellmeier_index` only supports formula 2.  Minimal
installs without `refractiveindex` raise `ImportError` on these
26 entries with a clear install hint.  v4.16.1 candidate: add
`_polynomial_index` evaluator.

## What's new in 4.15.5

**Closes the v4.15.4 audit (`docs/audits/AUDIT_V4_15_4_2026_05_19.md`)
through P3.**  0 P0 + 4 P1 + 6 P2 + 5 P3.  Three of the 4 P1s closed
via a **V6 dispatcher meta-pin walker refactor** — discovery now
keys off the function's first-positional-parameter name (via AST
inspection) rather than a hand-curated name-prefix list, plus the
walker now descends into class bodies.  The remaining P1 cluster
closed by fixing two user-facing pitfalls in v4.15.4's OPD plotting
functions.  1922 unit pass (up from 1858); 34/34 validation pass.

### Headline: V6 walker (first-positional-param-name discovery)

v4.15.3/4 used a hand-curated name-prefix filter (`apply_*` /
`propagate_*` / `richards_wolf_*` / `debye_wolf_*`); 30+ public
`__all__` functions with 2-D `E`/`field`/`pupil` first positional
fell outside that filter.  v4.15.5 walks `__all__` and inspects
each function's `ast.arguments.args[0].arg` — if the name is in
`{'E', 'E_in', 'field', 'pupil', 'object_field', 'psf'}`, requires
`_check_2d_scalar_field` OR `_GUARD_EXEMPTIONS` entry.  Discovery
is grounded in the actual signature.

Plus class-body descent: `DeformableMirror.apply` was a 3-line
non-delegating phase-screen multiply hiding behind the walker's
v4.15.4 blanket class-method exclusion.  v4.15.5 guards it
directly and documents legitimate-delegators
(`ThinLens.apply`, `FreeSpace.apply`, etc.) in
`_DELEGATING_CLASS_METHODS`.

Walker entry-point count: **96 top-level + 3 class methods** (was
72 at v4.15.4).  39 guarded (was 25).  `_file_to_ast` cached →
walker wall time 1.5s → 0.03s warm (~30× speedup).

### P1 closures (4)

* `DeformableMirror.apply` 1-line guard + walker class-body descent
* 13 newly-guarded analyzers: `wave_opd_2d`, `M2`, `strehl_ratio`,
  `beam_d4sigma`, `coupling_efficiency`, `compute_psf/otf/mtf`,
  `encircled_energy_curve`, `koehler_image`, `extended_source_image`,
  `shack_hartmann`, `rays_from_field`, `resample_field`
* `plot_opd_fan` `fan_units='m'|'waves'` kwarg — closes the
  silent ~6e5 double-conversion trap when users pipe
  `opd_fan_data` output (which returns waves) into the plotter
  (which expected metres)
* `_radial_rms_profile` switched to centered RMS — matches the
  1-D fan + 2-D heatmap RMS convention.  Same OPD on the 4-panel
  summary now produces reconcilable numbers.  Example
  `plot_opd_summary_singlet.py` RMS now reports 0.8901 waves
  (was 1.3347 uncentered); PV unchanged at 2.9318 waves.

### P2 closures

* `_check_2d_scalar_field` gains `input_kind: str = 'field'`
  parameter — `richards_wolf_focus`, `debye_wolf_psf`,
  `compute_psf/otf/mtf` etc. get accurate "expected 2-D complex
  pupil" error messages.
* `plot_opd_summary` docstring corrected to explicitly state
  `opd_2d` is in metres.
* `plot_opd_summary` even-N central-row/col aligned via
  `(N - 1) // 2`.
* CHANGELOG v4.15.4 walker count refreshed (43→49 → actual 72 at
  v4.15.4 HEAD).

### P3 closures

`_file_to_ast` `@lru_cache` (~30× walker speedup); `plot_opd_fan`
docstring specifies "centered RMS"; `n_bins` exposed as
`radial_rms_n_bins='auto'` with auto-clamp for tiny grids; CHANGELOG
57-vs-63 reconcile; per-agent +1 footnote; wavelength annotation
fixed (587.56→633 nm).

### ROADMAP cleanup

6 items that shipped in v4.15.0/v4.15.1 moved from "v4.16 residual"
to "Shipped highlights" (polychromatic EE, vector Strehl,
resolution metrics, astigmatism mag+angle, OAP factory, Forbes
Q-type).  True v4.16 residual is now 2 items: V4 meta-pin
candidates + multi-process atomic-append for `storage.py`.

## What's new in 4.15.4

**Closes the v4.15.3 audit (`docs/audits/AUDIT_V4_15_3_2026_05_18.md`)
through P3 + adds two user-facing OPD plotting functions.**  0 P0 +
1 P1 + 6 P2 + 5 P3 — cleanest audit yield in the v4.15.x series.
1858 unit pass (up from 1822); 34/34 validation pass.

### Headline: walker scope refactor + 2 new OPD plotting functions

v4.15.3 shipped a structural counter-measure (`_check_2d_scalar_field`
helper + dispatcher meta-pin) to retire the recurring sibling-gap
meta-pattern.  But the walker scoped only to two packages — leaving
6 sibling entry points outside its view.  v4.15.4 makes discovery
`__all__`-based and broadens the name filter; the walker now
discovers **72 candidate entry points (25 guarded + 47 documented
exempt)** and guards the 6 newly-found siblings:

* `propagate_through_system_jax` (P1)
* `apply_dm`, `apply_detector` (in `analysis/`)
* `richards_wolf_focus`, `debye_wolf_psf` (vector diffraction)
* `apply_perturbations` documented exempt (first arg is a
  prescription dict, not a 2-D scalar field)

Plus a **fake-violation counter-pin** that injects a synthetic
unguarded function and asserts the meta-pin fires — making walker
correctness a positive signal rather than a count-floor heuristic.

### New OPD plotting functions

Two new public functions in `lumenairy.analysis.plotting` (matching
the visual style of the `OPDPy_Lumenairy_Crosscheck` figures):

* **`plot_opd_fan(py, opd_y, px, opd_x, *, wavelength=None,
  units='waves', ...)`** — 2-panel tangential + sagittal OPD fans.
  Inputs match `raytrace.opd_fan_data` return tuple.  Solid line +
  zero-reference + PV/RMS annotation.
* **`plot_opd_summary(opd_2d, dx, *, ...)`** — 4-panel: 2-D heatmap
  (delegates to `plot_wavefront`), radial RMS profile, tangential
  fan, sagittal fan.  Fan args optional — auto-extracts from 2-D
  OPD's central row/column when not provided.

Runnable example at `examples/plot_opd_summary_singlet.py`.

### P1/P2/P3 closures

* `_validation.py` lazy-import hoisted to module scope (~1 µs/call
  saved; the cited circular dep was hypothetical).
* `_PerturbedABCDFallbackSentinel` deleted (was dead code with only
  a docstring marker).
* 63 v4.15.3 tests previously failing under `-W error::DeprecationWarning`
  now pass (per-module `pytestmark` filter for the documented Schell
  `return_kind` warning).
* CHANGELOG corrections to the v4.15.3 entry: SAS-anamorphic wording
  ("regardless" → "when self.method=='auto'"); stacklevel breakdown
  acknowledges `create_led_source` as factory-not-classmethod;
  dispatcher meta-pin count 17 guarded / 26 exempt (was 18/25);
  test-count arithmetic reconciled with documented +1 attribution-
  vs-collection gap.
* ROADMAP refreshed: test count → actual 1824; AUDIT_V4_15_2 +
  AUDIT_V4_15_3 added to closed list; meta-pin coverage 3 of 5 →
  4 of 5.

## What's new in 4.15.3

**Closes the v4.15.2 audit (`docs/audits/AUDIT_V4_15_2_2026_05_18.md`)
through P3.**  1 P0 + 4 P1 + ~6 P2 + ~4 P3.  The P0 was the recurring
"fix N, miss N+1" sibling-gap meta-pattern; v4.15.3 fixes it
**structurally** with a shared validation helper + dispatcher meta-pin.
1822 unit tests pass (up from 1732); 34/34 validation files pass.

### Headline: structural counter-measure for the sibling-gap pattern

v4.15.2 guarded 10 propagator/lens entry points; this audit found 9
more siblings (`*_propagate_mft`, `*_propagate_tilted`,
`apply_spherical/aspheric/grin/axicon/real_lens_traced/real_lens_maslov`).
v4.15.3 closes all 9 AND ships:

* **`lumenairy/_validation.py`** with `_check_2d_scalar_field(E, fn_name)` —
  shared guard helper replacing ~240 LOC of duplicated boilerplate.
* **Dispatcher meta-pin** that AST-walks `propagators/` + `elements/`
  for `def apply_*` and `def *_propagate*` and asserts the helper is
  the first executable statement of each entry point.  43 entry
  points discovered, 18 guarded, 25 documented exemptions.  Adding a
  new entry point without the helper now fails CI.

This is the 5th structural meta-pin: cache-clear (v4.14.1),
cache-locks + 0+0j (v4.14.2), `_validate_grid_params` (v4.15.0),
`_check_2d_scalar_field` (v4.15.3).

### P1 closures (4)

* **`FreeSpace._apply` SAS-anamorphic crash fixed** — when `dy != dx`
  and `method='auto'`, the dispatcher routed to SAS which doesn't
  accept `dy`.  v4.15.3 forces `method='asm'` when `dy != dx`.
  `FourierTransform` inherits the fix by composition.
* **Schell `DeprecationWarning` stacklevel** corrected from 4 → 5;
  library-wide sweep bumped 6 additional Source classmethod
  shim warnings from 3 → 4.
* **3 dead `optimize/core.py` sentinels wired** at their callsites
  (or marked dead-code with `_v4_15_3_dead_code = True` for the
  tuple-shape case that didn't sentinel cleanly).
* **`Source.gaussian_schell` / `Source.schell_model` classmethods**
  now route through `_RETURN_KIND_UNSET` sentinel (previously
  bypassed the v4.15.2 DeprecationWarning).

### P2 closures

`_RETURN_KIND_UNSET` promoted to dedicated `_SchellReturnKindUnsetSentinel`
subclass; rays_from_field threshold-comparison consistency (all 3
placement modes now `>=`); non-tautological FourierTransform pin via
Gaussian-beam waist relation `w_out = lambda*f/(pi*w_in)` (Saleh & Teich
§3.2.2, <0.0001% measured error); 4-fold mirror Cassegrain folded-
prescription test cases; library-wide `_warn_*` stacklevel sweep.

### P3 closures

CHANGELOG Forbes Q OPD bullet corrected (tolerance 1e-3 → 5e-3;
formula gains `(n-1)` factor); sentinel-migration line citations
refreshed post-drift; test-count reconciliation across pytest,
CHANGELOG, and ROADMAP.

## What's new in 4.15.2

**Closes the v4.15.1 audit (`docs/audits/AUDIT_V4_15_1_2026_05_18.md`)
through P3.**  1 P0 + 9 P1 + ~12 P2 + ~10 P3 — most were downstream-
integration gaps from the rapid v4.15.1 expansion (new types shipped
without updating consumers; breaking changes shipped without CHANGELOG
flagging; primitive APIs asymmetric).  1732 unit tests pass (up from
1625); 34/34 validation files pass.

### Breaking changes (with one-release deprecation warning)

* **Schell-family factories** now emit `DeprecationWarning` on the
  default `return_kind` path.  v4.15.1 silently changed the return
  shape from `(E_2d, x, y)` to `(E_3d, dx, dy, wavelength)` 4-tuple
  without warning.  v4.15.2 requires callers to pass
  `return_kind='ensemble'` or `return_kind='mcf'` explicitly;
  `version_removed='5.0'`.
* **`Source.gaussian_schell` / `Source.schell_model`** classmethods
  now return the factory's 4-tuple verbatim instead of wrapping the
  3-D ensemble in a `Source` (which broke the canonical 2-D
  `Source.E` contract).

### P0 closure

`_RETURN_KIND_UNSET` sentinel detects the Schell default-path entry;
`_warn_schell_return_kind_default` helper routes through the
canonical `_deprecation.warn_deprecated_signature` machinery.

### P1 closures (9)

* **`FourierTransform` 3-stage rewrite** — `_apply` now runs the
  literal `FreeSpace(f) -> ThinLens(f) -> FreeSpace(f)` chain so the
  ABCD claim and the field implementation finally agree (Goodman §5.2).
  ~2x slower than v4.15.1's 2-stage shortcut; users wanting hardware-
  realistic back-focal-plane semantics can compose `ThinLens(f) *
  FreeSpace(f)` directly.
* **`from_prescription` flat-mirror parity** now matches `system_abcd`
  (only curved mirrors flip parity).  Folded-prescription test cases
  pin the 1e-12 ABCD parity for folded systems.
* **`FreeSpace._apply` threads `dy`** — anamorphic chains
  `Magnify(a_x, a_y) * FreeSpace(d)` no longer silently propagate on
  the wrong grid.
* **`rays_from_field` `'cdf'` placement pixel-wise threshold** —
  consistent with `'rejection'` and `'uniform'` modes.
* **10 propagator entry points guarded** against `PartialCoherenceMCF`
  (clear `TypeError` with v4.16+ scope message) and 3-D ensemble
  shape (clear `ValueError` with iterate-over-ensemble pattern):
  `propagate_through_system`, `propagate` dispatcher, 5 wave-propagator
  entry points, 3 lens entry points.
* **CHANGELOG `### Breaking changes` subhead** added to v4.15.1
  entry retroactively (Schell return / `strehl_vector` default ref /
  `system.evaluate` mixed-shape).
* **`rays_from_field` short-return `RuntimeWarning`** when
  `n_actual < n_rays`; `n_rays = 0` early-return honoured.
* **ROADMAP refresh** to v4.15.2 / 1732-test baseline.

### P2 closures

* `_sentinel_unpickle` fallback raises clear `ImportError` for
  unknown subclasses (distributed-pipeline safety).
* **3 additional `optimize/core.py` sentinels** migrated to inherit
  from `_deprecation._Sentinel`: `_InvalidFocalLengthSentinel`,
  `_FailedScanStrehlSentinel`, `_PerturbedABCDFallbackSentinel`.
  `_NO_DEFAULT` promoted to dedicated `_NoDefaultSentinel` subclass.
  All registered for pickle round-trip safety.
* **`PartialCoherenceMCF.coherence_at` Hermiticity test** added.
* **UI runtime test** under `-W error::DeprecationWarning` —
  catches the static-grep escape class.
* **`rays_from_field` top-of-file docstring** Madelung → phase-ratio.
* **`Magnify` docstring direction** inverted to match code
  (Nazarathy/Shamir `V[a]` convention: `a > 1` shrinks).  Dead
  `operators.py:556-577` reference removed.
* **`'uniform'` + `'unwrap_gradient'` modes** in `rays_from_field`
  gain direct test coverage (vortex direction-recovery, anamorphic
  direction-cosines, cdf reproducibility).

### P3 closures

Sparrow tolerance pin tightened 5% → 1% (measured 0.017% on canonical
Airy); Forbes Q-bfs end-to-end OPD analytical pin; `lumenairy.algebra`
exports moved from Tier-2 to Tier-1 (build-time surface);
CHANGELOG line-citation refreshes; `energy_threshold` forwarded
through all 3 Schell factories; stray `C:tmpsources_diff.txt`
(44 KB git-diff dump) deleted.

## What's new in 4.15.1

**Closes the v4.15.0 audit (`docs/audits/AUDIT_V4_15_0_2026_05_18.md`)
through P3 + ships 2 additive features from
`docs/audits/CLUSTER_B_SPEC.md`.**  The audit was the highest-yield
in the series (2 P0s + 12 P1s + many P2/P3 on the +13 kLOC v4.15.0
release).  v4.15.1 closes both P0s + all Tier-0 P1s + the P2/P3
sweep + adds 800+ LOC of new CLUSTER_B surface.  1625 unit tests
pass (up from 1425); 34/34 validation files pass.

### P0 closures

* **`make_off_axis_parabola` factory fix (doubly broken).**  Decenter
  formula corrected `f*tan(alpha) -> 2*f*tan(alpha)` (chief-ray
  geometry on parent paraboloid).  Tilt 3-tuple convention pinned to
  `apply_real_lens_traced` — docstring loudly warns against use with
  paraxial `apply_real_lens`.  New end-to-end raytrace test pins
  off-axis focal-point location to 1% at 45° fold.
* **Schell-family redesign — hybrid ensemble + MCF.**  v4.15.0
  collapsed the `n_realizations` ensemble into a single coherent
  field before return; the partial-coherence contract was
  unfulfillable.  v4.15.1 returns the `(n_realizations, Ny, Nx)`
  ensemble by default (caller iterates + averages intensities for
  physically-correct partial coherence) plus an opt-in
  `return_kind='mcf'` mode returning a new `PartialCoherenceMCF`
  dataclass with coherent-mode decomposition (Wolf 1982 JOSA).
  Random-phase RMS-normalization bug fixed:  Schell-kernel
  `sigma_g` now actually controls coherence length
  (`sigma_g -> 0` recovers incoherent; `sigma_g -> infinity`
  recovers coherent).

### P1 closures (10)

* `sparrow_resolution` rewritten to canonical two-source dip-vanishing
  root-finding (was 15% low on Airy; now matches analytical
  `0.947 * lambda * f/#` to <5%).
* 7 UI Source-factory deprecation callsites migrated to kwarg-only
  form (the v4.15.0 release that introduced the deprecation now
  also migrates its own internal UI consumers).
* Forbes Q consumer paths closed:  raytrace flat-keys allowlist
  extended with `q_bfs_coeffs`/`q_con_coeffs`/`r_max`; Zemax `.zmx`
  QBFS/QCON SURFTYPE parsing added; wave-optics `apply_real_lens`
  now properly applies Q-bfs/Q-con sag (previously silently
  warning-and-skipped); Q-bfs/Q-con rectangular clip replaced
  with radial `r <= r_max` clip; `surface_sag_freeform` now
  requires `r_max` for q_bfs/q_con (eliminates the silent
  `r_max=1.0` unit-mismatch).
* `strehl_vector` default-reference removed (breaking) — produced
  unity for any uniform field of equal power AND Strehl > 1 on
  focused PSFs; now requires explicit `reference=`.
* `rayleigh_resolution` Gaussian-PSF false-positive fixed —
  Gaussian-like PSFs (no true zero) now return NaN with
  `RuntimeWarning`.
* `astigmatism_mag_angle` docstring range correction
  (`(-pi/4, pi/4]` -> `(-pi/2, pi/2]`).
* `system.evaluate` mixed-shape prescription now raises `ValueError`
  rather than silently picking a schema (small behaviour change for
  Zemax-loader output passthrough).

### P2 closures

* `__all__` symmetry — `surface_sag_q_bfs`, `surface_sag_q_con`
  re-exported from `lumenairy/elements/__init__.py`;
  `make_off_axis_parabola` from `lumenairy/io/__init__.py`.
* Sentinel consolidation — `_ZeroApertureMaskSentinel` and
  `_AngleUnsetSentinel` now inherit from `_deprecation._Sentinel`
  base; pickle-safe `__reduce__` + name-keyed registry so
  unpickling returns the singleton instance.
* `system.evaluate` Zemax-shape round-trip test added.

### P3 closures

`n=1.0` consistency between `_resolve_lens_glass_index` and
`register_fixed_glass`; codegen major-version-bump `UserWarning`;
LambertianBSDF surface-frame `RuntimeWarning`; coatings TIR-cap
always-emit warning; Forbes Q orthonormalizer docstring formula
correction; line-citation drift refresh.

### CLUSTER_B Item 6 — `rays_from_field` bridge

New top-level `lumenairy.rays_from_field(E, *, dx, wavelength, ...)
-> RayBundle`.  Samples a coherent field into a geometric ray
bundle for hybrid wave/ray analysis.  3 placement modes (cdf,
rejection, uniform); 2 angle methods (complex_gradient,
unwrap_gradient).  Bridges `propagators/` <-> `raytrace/` for users
who want to overlay ray traces on coherent-field plots, seed a
Maslov/GBD bundle from a measured pupil field, or hand a coherent
field into the geometric ray tracer.

### CLUSTER_B Item 2 — Operator algebra

New `lumenairy/algebra/` subpackage implementing Nazarathy/Shamir
operator algebra (JOSA 70 (2), 1980).  9 new top-level symbols:
`Operator`, `CompositeOperator`, `FreeSpace`, `ThinLens`,
`CylindricalLens`, `Magnify`, `FourierTransform`, `Aperture`,
`GaussianAperture`.  Build optical systems symbolically with `*`
composition; ABCD readable without applying to a field; or apply
to a `Source` to walk the chain.  `Operator.from_prescription(...)`
factory produces a `CompositeOperator` whose ABCD matches
`system_abcd(...)` to 1e-12 absolute.  Paraxial-only — for high-NA
prescriptions use `propagate_through_system` directly.

## What's new in 4.15.0

**Major minor release** rolling together carryover P1s from the
v4.14.2 audit + ROADMAP v4.15 + ROADMAP v4.16 into a single
coordinated ship.  1425 unit tests pass (up from 1265); 34/34
validation files pass.

### Headline: 19.4x modal-asymptotic perf win + wrong-saddle fix

`propagate_modal_asymptotic` switches to a single batched cold-start
`_solve_envelope_stationary_batch` + `_compute_M_b_batch` path.
Closes the v4.14.0 audit's "wrong-saddle basin" finding (the warm-
start chain entered wrong-saddle basins at grid edges, silently
zeroing those pixels via the `|b_quad| > 700` overflow guard).
**Output is bit-different from v4.14.x** at grid edges — strictly
more non-zero pixels because the cold-start finds the physical
saddle uniformly.  Four bit-equal pins migrated to property pins
(`1e-8` abs vs cold-start ref + 5% energy + nz-count >= warm-start);
the v4.14.1 row-reset pin retargeted to the v4.15 stronger
structural guarantee.  Measured 19.4x at N=128 LG_(0,0).

### New public API (10 functions)

* **Analysis (7):** `ee_polychromatic`, `strehl_vector`,
  `coupling_efficiency_vector`, `rayleigh_resolution`,
  `sparrow_resolution`, `fwhm_resolution`, `astigmatism_mag_angle`.
* **Sources (3):** `create_gaussian_schell_source`,
  `create_schell_model_source`, `create_annular_incoherent_source`
  (random-phase ensemble for partial coherence).  Matching
  `Source.gaussian_schell` / `Source.schell_model` classmethods.
* **System entry:** `lumenairy.system.evaluate(prescription,
  source, ...)` — accepts both Zemax-loader and factory-shape
  prescriptions; users loading a `.zmx` file no longer have to
  build the element list manually.
* **Freeform / geometry (3):** `surface_sag_q_bfs` and
  `surface_sag_q_con` (Forbes 2007 / Forbes 2010 orthonormal
  bases via shifted-Jacobi 3-term recurrence; orthonormality
  verified to <1e-6 for the first 5 orders); `make_off_axis_parabola`
  prescription factory.

### Source factory normalisation

5 `Source.*` classmethod factories migrated to canonical
`Source.method(*, N, dx, wavelength, <size_kwargs>)` (kwarg-only).
Legacy positional form works for one release with a
`DeprecationWarning` routed through the new
`_deprecation.warn_deprecated_signature` helper
(`version_removed='5.0'`).

### v4.14.2 carryover P1s closed

* **UI subpackage:** 7 findings fixed (glass-table N-LASF9 / S-NPH1,
  coordinate-mode-aware nudge, undo state-capture for
  wavelength/field weights + lens_options, back-vertex helper
  consolidation, waveoptics-dock re-parent shiboken guard, OPD
  mean-per-pixel accumulation, in-bounds mask before pupil
  indexing).
* **Codegen version pin** stamps `lumenairy_version` + a runtime
  `>= (4, 15, 0)` gate in generated scripts.
* **Ghost MC** accepts `seed: int | None = None` (default `None`
  for real uncertainty bands).
* **Glass Sellmeier bundle** for `SiO2` / `F_SILICA` /
  `FUSED_SILICA` / `S-LAH64` / `S-LAH79` so minimal installs
  without `refractiveindex` work.

### Exhaustive P2/P3 sweep + new meta-pin

User-requested exhaustive sweep of the v4.14.2 audit's 18 P2 + 12
P3.  7-8 of 18 P2 + 4 of 12 P3 closed; 10-11 P2 + 8 P3 deferred
(architectural for v4.16+ / v5.0).  New input-validation
entry-point meta-pin walks every `create_*` factory and asserts
`_validate_grid_params` in first 15 body lines (17 PASS + 1
documented `create_led_source` xfail).

### Version-stamping on HDF5 / Zarr writes

Every `create_dataset` / `create_array` / `create_group` site
now stamps `lumenairy_version` (7 locations).  Future-proof for
cross-version field-file compatibility.

### ROADMAP refresh

`ROADMAP.md` updated to v4.15.0 baseline.  v4.14.1 / v4.14.2 /
v4.14.3 / v4.15.0 shipped highlights added; closed items moved
out of v4.15 / v4.16 target lists; remaining items renumbered.

## What's new in 4.14.3

**Closes the v4.14.2 audit (`docs/audits/AUDIT_V4_14_2_2026_05_17.md`).**
The audit found 2 NEW P0s + 21 P1s + 18 P2 + 12 P3.  The
"fix N, miss N+1" sibling-gap meta-finding recurred 5 ways on
v4.14.2.  **v4.14.3 closes both P0s + the 5 sibling-gap
recurrences + 11 latent-bug P1s (including 1 real physics error
in `multiconfig.py`) + all 3 doc-drift P1s.**  1265 unit tests
pass (up from 1190); 34/34 validation files pass.

### P0 closures

* **`storage.py` `n_planes` atomicity.**  `append_plane_h5` and
  `_zarr_append_plane` previously bumped `n_planes` AFTER
  `create_dataset`; an interrupt between the two left an orphan
  dataset that the next append silently clobbered on the Zarr
  path (`overwrite=True`).  Concurrent appenders could race on
  `n_planes=N` and both write to `plane_{N:02d}`.  v4.14.3
  inverts the ordering (attr first, dataset second, try/except
  rollback) and drops `overwrite=True` on the Zarr path so
  orphan re-use now raises rather than silently destroys data.
  Multi-process locking deferred to v4.15+ (documented
  restriction).
* **`makedammann2d` >1m upper-bound** plus explicit
  `_legacy_units='auto'|'um'|'SI'` opt-in kwarg.  v4.14.2's
  `value > 1e-3` heuristic silently mangled legitimate mm-scale
  SI gratings (coarse industrial Dammann, THz/MMW) by 1e-6.
  v4.14.3 rejects any unit-bearing input > 1.0 m in `'auto'` /
  `'SI'` mode with a clear `ValueError` and provides explicit-
  `_legacy_units` escape hatches in both directions.

### P1 closures — sibling-gap recurrences (5)

* **`clear_asm_caches()` LG-polynomial chain** — adds the 8th
  cache (`_lg_polynomial_items`) v4.14.2 missed.
* **`apply_rotator` symmetry across 5 polarization helpers** —
  `_AngleUnsetSentinel` + a single `_resolve_angle` helper
  used by `apply_polarizer`, `apply_waveplate`,
  `apply_half_wave_plate`, `apply_quarter_wave_plate`, and
  `apply_rotator`.  The explicit-`angle=0` + `angle_deg=90`
  conflict (which v4.14.2's `angle != 0.0` check silently let
  through) now raises uniformly.
* **`create_led_source` legacy-shim scale-inversion check** —
  the canonical-order positional mistake that previously
  silently mis-routed `wavelength` into the `diameter` position
  now raises `TypeError`.
* **`_validate_grid_params` `support_tuple_N: bool` gating** —
  the 3 factories with genuine 2-D grid support opt in; the
  other 7 reject tuple-N at validation time with a clear
  `TypeError` instead of an opaque downstream `np.arange`
  crash.

### P1 closures — under-examined modules (6)

* **`multiconfig.py` hardcoded `n=1.5` (REAL physics error)** —
  multi-config beam-expander / Keplerian-telescope builders
  computed lens powers via lensmaker formula with a hardcoded
  BK7-ish index, producing wrong focal lengths for any non-BK7
  glass.  v4.14.3 routes through `glass.get_glass_index()`
  with documented `UserWarning` + bounded fallback for unknown
  labels.
* **`create_bessel_beam` cone_angle constraint** (`0 < a < pi/2`).
* **`create_fiber_mode` `mode_field_diameter > 0`** constraint.
* **`surface_sag_xy_polynomial` / `surface_sag_chebyshev`
  negative-`norm_x`/`norm_y` rejection** at function entry.
* **`ghost.py` IndexError on `elements`-key prescriptions** —
  new `_ghost_surfaces` adapter routes both `elements`-style
  schemas through the canonical surface-expansion path.
* **`register_fixed_glass` input validation** — non-empty
  string name, finite `n` in `[1.0, 4.0]`, `UserWarning` on
  overwrite.

### P1 closures — doc-drift (retroactive to v4.14.2)

3 retroactive corrections promised in v4.14.2 but not delivered
(Agent D test count `25` → `8`; `6 factories` → `5` in 3
sites; `lenses_maslov.py` line drift refreshed
`678/737/884/993` → `619/728/874`).  11 `AUDIT_V4_*.md`
references now carry the `docs/audits/` prefix.  Cache-locks
meta-pin count corrected `38` → `39 (38 pass + 1 documented
skip)`.

## Cookbook

Short, self-contained examples for the six public functions
added in v4.14.0.  Each block is a runnable snippet (assumes a
prior `import lumenairy as la` and a populated complex field
`E` on an `N x N` grid with pixel pitch `dx`).

### Encircled energy curve

`encircled_energy_curve(E, dx, *, dy=None, radii=None,
centroid=None, n_radii=64)` returns the fraction of total power
within radius `r` of the centroid as a 1-D curve.

```python
import numpy as np
import lumenairy as la

N, dx = 256, 2e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
E = np.exp(-(X * X + Y * Y) / (2 * (10e-6) ** 2)).astype(np.complex128)

radii, ee = la.encircled_energy_curve(E, dx, n_radii=64)
print(f"50% encircled radius ~= {radii[np.searchsorted(ee, 0.5)] * 1e6:.2f} um")
```

### Encircled energy radius

`encircled_energy_radius(E, dx, *, dy=None, threshold=0.84,
centroid=None)` returns the radius enclosing the given
threshold fraction (default 0.84 -- the standard "84% encircled
radius" lens-spec convention).

```python
import numpy as np
import lumenairy as la

N, dx = 256, 2e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
E = np.exp(-(X * X + Y * Y) / (2 * (10e-6) ** 2)).astype(np.complex128)

r84 = la.encircled_energy_radius(E, dx, threshold=0.84)
print(f"r84 = {r84 * 1e6:.2f} um")
```

### MTF cutoff

`mtf_cutoff(mtf_profile, freq, *, threshold=0.5)` returns the
spatial frequency at which a 1-D MTF profile drops below
`threshold`.  Returns `np.inf` if the MTF stays above the
threshold for all frequencies.

```python
import numpy as np
import lumenairy as la

# Synthetic Gaussian MTF profile.
freq = np.linspace(0, 1000, 200)  # cycles/mm
mtf = np.exp(-(freq / 300.0) ** 2)
f50 = la.mtf_cutoff(mtf, freq, threshold=0.5)
print(f"50% MTF cutoff = {f50:.1f} cycles/mm")
```

### Beam diameter

`beam_diameter(E, dx, *, dy=None, threshold='1/e^2',
centroid=None)` returns the diameter at which the intensity
drops below the named threshold.  Accepts `'1/e^2'`, `'1/e'`,
`'FWHM'`, or `'D4sigma'`.

```python
import numpy as np
import lumenairy as la

N, dx = 256, 2e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
E = np.exp(-(X * X + Y * Y) / (2 * (10e-6) ** 2)).astype(np.complex128)

d_1e2 = la.beam_diameter(E, dx, threshold='1/e^2')
d_4s = la.beam_diameter(E, dx, threshold='D4sigma')
print(f"1/e^2 diameter = {d_1e2 * 1e6:.2f} um; D4sigma = {d_4s * 1e6:.2f} um")
```

### Depth of focus

`depth_of_focus(wavelength, f_number, *, formula='rayleigh')`
returns the one-sided depth of focus.  `'rayleigh'` evaluates
`+/-4 f# ** 2 * lambda`; `'marechal'` evaluates `+/-lambda /
NA ** 2`.

```python
import lumenairy as la

# F/2.8 lens at 633 nm.
dof = la.depth_of_focus(wavelength=633e-9, f_number=2.8,
                        formula='rayleigh')
print(f"Rayleigh depth of focus = +/-{dof * 1e6:.2f} um")
```

### Wavefront map plot

`plot_wavefront(opd, dx, *, dy=None, aperture=None,
units='waves', wavelength=None, cmap='RdBu_r',
show_stats=True, ax=None, fig=None, title=None)` produces a
Zemax-style wavefront map: NaN outside the aperture, divergent
colormap centred on zero, PV/RMS overlay.

```python
import numpy as np
import lumenairy as la

N, dx = 128, 4e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
# Synthetic OPD: 1/2 wave of defocus across a 200 um aperture.
opd = 0.5 * 633e-9 * (X * X + Y * Y) / (100e-6) ** 2
aperture = (X * X + Y * Y) <= (100e-6) ** 2

# Headless backend so the example runs in CI without a display.
import matplotlib
matplotlib.use('Agg')
fig, ax = la.plot_wavefront(opd, dx, aperture=aperture,
                            units='waves', wavelength=633e-9,
                            title='Defocus')
```

### Anamorphic-grid (`dy`) migration example for `makedammann2d`

A short worked example for users migrating off the legacy
`makedammann2d` µm-units form to the v4.14.3+ explicit
`_legacy_units='SI'` opt-in.  Required reading for THz / MMW
DOE designers with mm-scale periods.

```python
import lumenairy as la

# Pre-v4.14.3 (legacy) -- silent µm-units rescale at value > 1e-3.
# T = la.makedammann2d(periodx=100, periody=100, waveln=1.31)

# v4.14.3+ canonical SI form -- explicit opt-in, no rescale.
T = la.makedammann2d(periodx=100e-6, periody=100e-6,
                     waveln=1.31e-6, _legacy_units='SI')
```

## What's new in 4.14.2

**Closes the v4.14.1 audit (`docs/audits/AUDIT_V4_14_1_2026_05_17.md`).**
The audit found 1 NEW P0 (a 3-release carryover: `glass.py`
S-LAH64/S-LAH79 dispatch broken since v4.11.2) + 10 new P1s,
including 4 "fix N, miss N+1" recurrences on v4.14.1 itself
(aperture=0 sentinel missed 2 sites; 7 older caches still unlocked;
4 residual `0+0j` literal sites; `clear_asm_caches` chain narrower
than docstring) and 6 P1s in under-examined modules (freeform
domain guard, makedammann unit drift, polarization API gaps,
source factory validation).  **v4.14.2 closes the P0 + all 10 P1s
+ a follow-up cache-lock gap discovered by Agent C's new meta-pin.**
1190 unit tests pass; 34/34 validation files pass.

### Documentation reorganisation

* `docs/audits/` — all audit reports and `CORRECTION_PLAN.md` moved
  from repo root.
* `docs/release_notes/` — all `.release_notes_v*.md` per-release
  drafts moved from repo root.

Repo root now contains only the primary project docs:
`README.md`, `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md`,
`GUI_README.md`, `GUI_CHANGELOG.md`, `REAL_LENS_CHANGES.md`.

### P0 closure — `glass.py` S-LAH64 / S-LAH79 dispatch

3-release carryover regression.  v4.11.2 removed both glasses from
`SELLMEIER_COEFFICIENTS` after discovering the in-code coefficients
were off by ±5.8% vs the Ohara catalog — but **never updated
`GLASS_REGISTRY`**, leaving both flagged `'__sellmeier__'`.  The
dispatcher raised `ValueError` on every call.  UI's "known-good
preset" path broken for 3 releases.

v4.14.2 routes both to `('specs', 'OHARA-optical', '<name>')` tuple
form (catalogue book name verified by introspection).  Numerical
agreement vs Ohara catalog: 5e-5.  Module-load consistency check
added that fails fast at import time on any future
`GLASS_REGISTRY` ↔ `SELLMEIER_COEFFICIENTS` drift.

### P1 closures — sibling-gap recurrences on v4.14.1

* **Aperture=0 sentinel finish** — v4.14.1 missed `ToleranceAware
  Merit._evaluate_perturbed` and `MatchIdealSystem._make_source`.
  Both now use the canonical `is _ZERO_APERTURE_MASK` branch.
  Investigation finding: `apply_perturbations` doesn't mutate
  prescription-level `aperture_diameter` (only per-surface
  `decenter`/`tilt`/`form_error`), so the audit's worry about
  perturbed-to-zero apertures is not triggered — pinned the
  invariant as a regression guard.
* **7 older caches now locked** (`_ZERNIKE_BASIS_CACHE`,
  `_THROUGH_FOCUS_SCAN_JAX_CACHE`, `_PROPAGATE_SYSTEM_JAX_CACHE`,
  `_GS_KERNEL_CACHE`, `_ER_KERNEL_CACHE`, `_HIO_KERNEL_CACHE`,
  `_TRACE_JAX_CACHE`).  Lock-scope discipline matches v4.14.1
  (hold for `OrderedDict` ops only, release before expensive
  kernel build).  Concurrent 4-thread tests confirm no
  exceptions.
* **`clear_asm_caches()` chain expanded** to 5 sibling caches via
  lazy-import + call.  Docstring rewritten to honestly enumerate
  all caches it clears.
* **2 P1-severity residual `0+0j` literal sites** swept
  (`optimize/core.py:966`, `phase_retrieval.py:402`).

### P1 closures — under-examined modules

* **`surface_sag_xy_polynomial` domain guard** added (matches the
  Chebyshev-branch precedent).
* **`makedammann2d` SI metres** with per-parameter deprecation
  heuristic — handles pure-legacy and hybrid µm/SI calls without
  breaking existing tests.
* **`apply_rotator` accepts `angle_deg=`** (matches v4.7
  polarization-family convention).
* **`JonesField.__init__` input validation** (2-D shape +
  positive-finite `dx, dy`).
* **`create_led_source` signature normalised** to v4.7-canonical
  `(N, dx, wavelength, *, diameter, divergence_angle, dy=None, ...)`
  with legacy-positional deprecation shim.
* **`_validate_grid_params` helper** + applied to all 10 `create_*`
  factories in `sources/core.py`.

### Follow-up cache-lock gap closed

The new cache↔lock meta-pin discovered `_JAX_IFT_SOLVER_CACHE` in
`propagators/asymptotic.py` — a single-cell lazy-init cache
without a lock.  Race window on first concurrent call would
double-decorate the JAX `custom_vjp` solver.  v4.14.2 closes the
gap in the same release via double-check locking pattern:
fast-path no-lock for the common populated case; slow-path
acquires lock, re-checks, delegates the actual build to
`_build_jax_ift_solver_impl` (worker function).  Meta-pin's
known-gap set is now empty.

### Two new structural meta-pins

Extending v4.14.1's cache-clear dispatcher-pin pattern to two
more sibling-gap classes:

* **`test_v4_14_2_dispatcher_pin_cache_locks.py`** — 39 collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK` skip):
  every module-level `_<NAME>_CACHE` must have a corresponding
  lock.  Accepts both `_FOO_LOCK` and `_FOO_CACHE_LOCK` naming
  conventions.  Reverse check on `_LOCK` names + documented
  `_PATCH_LOCK` exemption.
* **`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`** (123 tests)
  — walks all 117 `lumenairy/*.py` files for
  `np.where(.*, 0+0j)` literals.  Three exemption layers
  (comment lines, trailing `.astype()` recovery, explicit P3
  allowlist).

## What's new in 4.14.1

**Closes the v4.14.0 audit (`docs/audits/AUDIT_V4_14_0_2026_05_17.md`).**
v4.14.0 shipped 7 perf wins + 6 new public functions + 80
parametrized dispatcher pins, but the audit found 1 P0 silent-
wrong-physics bug in the 77× LG/HG mode-stack cache key plus 6
P1s.  **v4.14.1 closes the P0 + all 6 P1s + the top-priority P2s.**
911 unit tests pass (up from 858); 34/34 validation files pass.

### P0 closure — LG/HG mode-stack cache key

The 77× perf win in v4.14.0 had a silent-wrong-physics bug.  The
cache key omitted `dx, dy` — two calls with the same shape but
different physical pitch (e.g. `dx=1e-6` then `dx=2e-6`, both at
N=256) **collided on this key**, returning the FIRST grid's modes
evaluated against the second call's field.  Silently wrong on
multi-resolution analysis, wavelength-adaptive grid sweeps, and
optimisation loops where `dx` is a free variable.  v4.14.1 adds
`dx, dy` to both keys with a regression pin.

### P1 closures

* **Aperture=0 semantics regression fixed.**  v4.14.0's wrapper-
  merit cache mapped `aperture <= 0` to `mask=None`, which downstream
  interpreted as "no clipping, full grid plane wave."  v4.14.1 adds
  a `_ZeroApertureMaskSentinel` so callers can distinguish "no
  aperture specified" from "aperture explicitly zero (block all
  light)."  Pre-v4.14 semantics restored.
* **Brewster-angle phase aggregation bug** in `coatings.py` (the
  v4.13.1 audit flagged it; v4.14.0's wavelength-batch rewrite
  inherited the bug).  `0.5 * (angle(r_s) + angle(r_p))` is wrong
  by π/2 or π at Brewster.  v4.14.1 changes to `np.angle(0.5 *
  (r_s + r_p))` (complex sum then angle — robust to π
  discontinuities).
* **`_solve_envelope_stationary_batch` contract violation** —
  function promised `converged=False` for failed pixels but set
  `True` to drop them from the active set.  Separate `finished`
  mask added; `converged` now matches the docstring.
* **`clear_lg_mode_stack_cache` now in top-level `__all__`** —
  v4.14.0's CHANGELOG claimed this was "Public" but the import
  wasn't in `lumenairy/__init__.py`.  The audit-meta-finding
  recurring on the very release that shipped 80 dispatcher pins.
  v4.14.1 closes the gap AND adds a structural counter-measure: a
  **cache-clear dispatcher pin** that walks every submodule's
  `__all__` for `clear_*` names and asserts each is re-exported
  at top level.  Future cache-clear additions can't regress this.
* **`encircled_energy_radius` docstring corrected** — claim
  `ee[0] = 0 always` is false; `ee[0]` equals the centre-pixel
  intensity contribution.

* **`row_reset` resets the Newton warm-start (P1-NEW-5)** — the
  `row_reset` branch now resets `last_v_star = (v_cx, v_cy)` at
  each raster row wrap, eliminating the cross-row Newton chain
  that plausibly entered wrong-saddle basins near grid edges.
  Coordinated with the fix, the bit-equal pin and the older
  1e-10 rel pins both had their inline scalar references updated
  to reset `last_v_star` at row wrap too.

### Tier-2 follow-ups

* **Thread-safety locks** on the three new v4.14.0 caches.
* **Monkey-patch removed** in `optimize/core.py` — `clear_asm_caches()`
  now lazy-imports and calls `_clear_wrapper_merit_cache` +
  `clear_lg_mode_stack_cache` directly, eliminating re-import
  recursion risk.
* **LG/HG mode-stack cache now wired into `clear_asm_caches()`**
  (v4.14.0 only wired it into `lumenairy_context()`).
* **Final `0+0j` / `1+0j` literal sweep** — 2 missed sites
  (`lenses_maslov.py:448`, `_lens_thin.py:173`).
* **`fiber_mode` added** to `TestP1CSourceFactoryDispatcherPin`
  parametrize list (stale exclusion from v4.13.2).

### Doc-trust hygiene

4 confirmed doc-drift items from the audit retroactively
corrected in the v4.14.0 CHANGELOG entry below: "16 tests at 1e-10
rel" (→ 3 in cited class), "6 batched helpers consumed by 77×
win" (→ helpers have zero production consumers, reserved for
v4.15+), HG cache key `w[s]` shorthand (→ both `wx`, `wy`),
`lenses_maslov.py` line drift.

## What's new in 4.14.0

**Phase B of the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`).**
v4.13.2 closed Tier-0; v4.14.0 closes Tier-1: 7 perf wins, 6 new
user-facing API functions, and 80 dispatcher pins that close out
the recurring sibling-gap audit-meta-finding.  All 858 unit tests
pass; 34/34 validation files pass.

### Performance wins

| Hot path | Speedup |
|---|---|
| `coating_reflectance` wavelength batch (50 layers × 200 wv) | **24.6×** |
| `decompose_lg` / `decompose_hg` cache (warm, 256², 28 modes) | **77×** |
| `MultiWavelengthMerit` meshgrid cache (N=512, 5 wl, 20 FD) | **6.17×** |
| `_evaluate_polynomial_4d_and_grad34` (typical configs) | **4.6×** |
| `MultiWavelengthMerit` meshgrid cache (N=128, 3 wl, 5 FD) | **4.16×** |
| `ToleranceAwareMerit` meshgrid cache | **3.17×** |
| Shack-Hartmann scatter-gather loop | **2.27×** |
| Phase-retrieval `angle`/`exp` round-trip (GS) | **2.26×** |
| Phase-retrieval (ER / HIO) | 1.85× / 1.59× |

`Multi*Merit` meshgrid build count drops from `O(n_wl × n_field ×
FD)` to **1 per `(N, dx, aperture)` signature** for an entire
optimisation run.  LG/HG mode-stack cache wired into
`clear_asm_caches()` and `lumenairy_context()`.

**Coating Snell-chain hoist** is what unlocks the 24× batched
speedup: the documented `n.imag-dropped-at-Snell-step` approximation
makes the per-layer chain wavelength-independent, walked ONCE
outside the polarisation loop.

**Modal asymptotic vectorisation** (audit target 20-100×) was
investigated but NOT shipped publicly — the cold-start batched
Newton finds the physical saddle uniformly across pixels, while the
pre-v4.14 warm-started Newton lands in wrong-saddle basins at grid
edges that the overflow guard zeros.  The existing test pin bit-
equals the (subtly wrong-at-edges) warm-start behaviour.  This is a
**real physics finding** worth a coordinated v4.15+ release that
updates the test pin alongside the algorithm change.  The
vectorised helpers ship privately and are consumed by the
`decompose_lg`/`decompose_hg` cache (the 77× warm win).

### New user-facing API

Six new functions exposed at `lumenairy.*`:

* `encircled_energy_curve(E, dx, *, dy=None, radii=None, ...)`
  returns `(radii, ee)` where `ee[i]` is the fraction of total
  power within `radii[i]` of the centroid.
* `encircled_energy_radius(E, dx, *, threshold=0.84, ...)` —
  standard "84% encircled radius" lens spec metric.
* `mtf_cutoff(mtf_profile, freq, *, threshold=0.5)` — spatial
  frequency at which a 1D MTF drops below threshold; returns
  `np.inf` if the MTF stays above threshold everywhere.
* `beam_diameter(E, dx, *, threshold='1/e^2', ...)` — diameter at
  intensity threshold.  String thresholds: `'1/e^2'`, `'1/e'`,
  `'FWHM'`, `'D4sigma'`.
* `depth_of_focus(wavelength, f_number, *, formula='rayleigh')` —
  Rayleigh / Marechal depth of focus.
* `plot_wavefront(opd, dx, *, dy=None, aperture=None,
  units='waves', wavelength=None, ...)` — Zemax-style wavefront
  map: NaN outside aperture, divergent colormap, PV/RMS overlay.

All six honour `dy` from the start; `dy=None` defaults to `dx`.

### Sibling-gap parametrized dispatcher pins (audit meta-finding closure)

80 new pins across 3 test files, closing out the recurring
"fix swept N sites, missed N+1" pattern for three sibling families:

* `(scalar, vectorial) HFPI` — 29 pins covering `_spawn_rng`
  independence, grazing-ray inf/NaN guards, alive-mask correctness.
* `(NumPy, JAX) apply_real_lens` family — 35 pins covering case-
  insensitive `glass_after='MIRROR'`, `dy` handling, dtype
  preservation across all 5 variants.
* Welford-mirror convention — 16 pins across `seidel_coefficients`,
  `petzval_radius`, `aberration_summary`, `chromatic_focal_shift`,
  `distortion_grid`, `field_aberration_sweep`,
  `eval_image_plane_wfe`.

**Sibling-gaps DISCOVERED and fixed in this release** by the new
pins (the meta-finding self-closing): complex64 dtype preservation
was broken on `apply_real_lens_maslov` (3 sites in
`lenses_maslov.py`) and both JAX twins (`_lens_jax.py:573, 819`).
Fixed by threading `out_dtype=E_in.dtype` through the maslov
helpers (`_integrate_quadrature`, `_integrate_local_quadrature`,
`_integrate_stationary_phase`) and routing `_resolve_jax_complex
_dtype(E_in.dtype)` instead of the library-default.  A final
post-normalisation `.astype(E_in.dtype)` catches the python-float
multiply that promotes complex64 → complex128.

## What's new in 4.13.2

**Closes the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`) plus its
Part 10 consolidation with a parallel cross-library survey.**  v4.13.1
audit identified 12 new P1s (5 sibling-gap recurrences + 5 fresh-eyes
bugs + 2 partial-closure follow-ups); the cross-library survey turned
up 5 additional P0s in `optimize/` + `io/` and 7 P1s mostly around
`dy` convention drift.  v4.13.2 closes the full consolidated Tier-0
set.  All 710 unit tests pass; 34/34 validation files pass.

### P0 closures (cross-library survey)

* **`make_lg_aberration_merit_jax`** no longer silently ignores its
  `targets` dict (loop body was literally `pass`; weighted sum was
  never computed).  Public API.  v4.13.2 weights the `(0,0)` target
  correctly; non-`(0,0)` raises `NotImplementedError`.
* **`MultiFieldMerit.field_angles`** accepts `(theta_x, theta_y)`
  tuples (was Y-axis tilt only despite docstring).  Scalar form
  preserved with a one-shot `DeprecationWarning`.
* **`load_plane_slice`** documented return type now matches actual
  (`(arr, attrs)` tuple).
* **CODE V `.seq` + Quadoa `.qos` round-trip preserves BFL** via a
  new `'back_focal_length'` prescription key populated by readers
  and exporters.

### Sibling-gap P1 closures

Five new "fix-swept-N-sites-missed-N+1" instances closed, each with
parametrized dispatcher pins:

* **`vectorial_hfpi` RNG-correlation** — v4.11.2 fixed scalar HFPI;
  the vectorial twin was missed.  Ports `_spawn_rng`.
* **`subaperture.propagate_subaperture_asymptotic`** — v4.11.2 added
  per-patch `decompose_lg` in `hf.py`; the subaperture sibling still
  passed unit-amplitude `(0,0)`.  **Functionally wrong for any non-
  trivial input.**  v4.13.2 ports the per-patch decomposition with
  3 new tunable kwargs.
* **`petzval_radius` Welford-mirror convention** — `seidel_coefficients`
  was fixed in v4.11.2; `petzval_radius` was still skipping every
  mirror.  **Wrong by 100% for catadioptric / Cassegrain designs.**
* **`_build_jax_prescription` `glass_after='MIRROR'`** check added
  (v4.13.1 P1-A's two-guard fix only partly applied to the JAX
  prescription builder).
* **JAX lens twins thread `dy=`** (`apply_real_lens_traced_jax`,
  `apply_real_lens_maslov_jax`).  Raise `ValueError` on `dy != dx`
  matching NumPy precedent.

### Fresh-eyes P1 closures

* **`apply_mirror` NaN guard** matches `apply_real_lens` (hyperbolic
  mirror at conic-domain boundary no longer poisons every pixel).
* **`_zero_C_air_gap`** raises `RuntimeError` on degenerate ABCD
  (was silently returning placeholder thickness).
* **`propagate_to_plane`** zeroes the step on dead/grazing rays in
  BOTH scalar (`hfpi`) and vector (`vectorial_hfpi`) paths.
* **`RandomState.choice` int dtype** aligned across NumPy and JAX
  backends (both return int64 now).
* **`trace_prescription`** uses `_surface_copy_with` instead of
  mutating shared `Surface.thickness` in place.

### Partial-closure follow-ups

* **`dual_annealing` callback wired into `CancellableProgress`**
  (v4.13.1's inline lambda didn't poll `is_cancelled`).
* **`RandomState.choice` old-JAX safety net** — `try/except TypeError`
  → `RuntimeError` with a clear "JAX >= 0.4.0 required" message.

### Cross-library survey P1 closures

* **`strehl_ratio` + `polychromatic_psf` accept `dy=`** (v4.13.0 L3
  sweep missed these).  Back-compat preserved bit-for-bit when
  `dy is None`.
* **Wrapper-merit context threads `x=ctx.x`** through the three
  `Multi*Merit` / `ToleranceAwareMerit` sub-contexts (the analytic-
  gradient path no longer silently degrades to FD inside these).
* **`propagate_through_system` element handlers thread `dy`**
  through all 13 element types (was silently squaring anamorphic
  `dy` to `dx`).
* **`Source.fiber_mode` accepts `dy=`** end-to-end (`create_fiber
  _mode` widened).  Dispatcher pin now covers all 6 factories.

### Thin-lens family sibling-gap sweep

Cross-library survey found additional sibling-gaps in the thin-lens
family:

* **9 sites of `0.0+0.0j` complex128 literal** replaced with
  `xp.zeros((), dtype=E.dtype)`.
* **6 thin-lens functions** now match `phase_exp.dtype` against
  `E.dtype` (were silently upcasting complex64 → complex128 via the
  `xp.exp(1j*phase)` mask).
* **3 thin-lens functions** gained the documented `use_gpu`
  parameter and canonical CuPy dispatch.
* **Latent CuPy dispatch bug fixed** in `apply_thin_lens`,
  `apply_spherical_lens`, `apply_aspheric_lens` — bare `cp`
  references (Python LEGB rules skip module-level PEP 562
  `__getattr__` for function-local lookups) routed through
  `_lenses_module.cp` instead.

### Quick wins

* 5 mis-tiered names in `__init__.py.__all__` moved to correct tiers.
* Duplicate `reset_fft_backend` import removed.
* Broken wiki TOC anchor fixed (`#whats-new-in-4-14-0` →
  `#whats-new-in-4-13-1`).

## What's new in 4.13.1

**Closes the v4.13.0 audit (`docs/audits/AUDIT_V4_13_0_2026_05_17.md`) plus an
additional perf-survey pass.**  v4.13.0 was tagged in git but never
published to PyPI; that audit identified 7 P1 (latent bug), 9 P2
(code smell), and 6 P3 (cleanup) findings.  v4.13.1 closes every
Tier-0 / Tier-1 / Tier-2 / Tier-3 item plus 3 new perf wins
discovered in the parallel survey.  All 654 unit tests pass;
34/34 validation files pass.

### Sibling-gap P1 closures

The audit's headline finding: 3 "sibling-gap" recurrences in
v4.13.0's own audit closures (a fix sweeps N call sites but
misses N+1).  All three closed in v4.13.1:

* **P1-A: `apply_real_lens` mirror guard** -- the v4.13.0 L4a
  sweep hardened 4 sibling `apply_real_lens_*` variants but
  missed the parent function itself.  A hand-built prescription
  with `surfaces[i]['is_mirror']=True` would silently miscompute
  through the parent.  Guard ported from the `_lens_traced.py`
  template; parametrized dispatcher pin added over all 5 variants
  to prevent future sibling-gaps in this family.
* **P1-B: ER/HIO complex-dtype routing** -- `gerchberg_saxton_jax`
  correctly routed `dtype` through `_resolve_jax_complex_dtype`
  in v4.13.0, but `error_reduction_jax` and `hybrid_input_output
  _jax` skipped the resolver -- the EXACT silent float64 ->
  complex64 demotion bug L2 was meant to close, still live.
  Both now route through the resolver; cache keys pinned on
  resolved complex dtype.  Parametrized pin across all 3 phase-
  retrieval kernels.
* **P1-C: `Source.propagate()` + 5 factories thread `dy`** --
  `Source.propagate(...)` and the 5 classmethod factories
  (`gaussian`, `plane_wave`, `point_source`, `top_hat`,
  `fiber_mode`) were dropping `dy` from the returned Source even
  though the underlying E-field was built on the anamorphic
  grid.  `Source.propagate` also gained an optional `output_dy`
  kwarg for symmetry with `output_dx`.  Parametrized dispatcher
  pin over the 4 dy-aware factories.

### UI + infrastructure P1 closures

* **P1-D: `ThinGratingDock` non-functional** -- the dock's `_run`
  was calling `grating_efficiency_vs_wavelength` with wrong
  kwargs (`groove_index`, `substrate_index`, `profile`, `angle`)
  while the function expected `(period, depth, *, n_ridge,
  n_groove, n_substrate, n_superstrate, order, ...)`.  Every
  click raised `TypeError` silently swallowed into the summary
  text box.  Rewrote with correct signature, added missing UI
  inputs.  Math path extracted into `_compute_efficiency_data
  (inputs: dict) -> dict` so it's unit-testable without Qt.
* **P1-E: `_context.py` cache-clear import moved inside try** --
  `from .propagators.propagation import clear_asm_caches` was
  OUTSIDE the try-block, so a rename / circular import there
  would bypass ALL 6 subsequent guarded cache-clears.  Fixed.
* **P1-F: `RandomState.choice` honours `replace=False` on JAX**
  -- previously silently ignored the flag on the `p=None` path.
  Now dispatches to `jax.random.choice(..., replace=False)`.
* **L7: benchmark cache clear** -- `test_bench_through_focus
  _scan_jax_first_vs_warm` now clears `_THROUGH_FOCUS_SCAN_JAX
  _CACHE` before timing, matching the 4 sibling benchmarks.

### Tier-2 follow-ups (P2 code smells)

* **`_RestoreDtype` try/finally** -- explicit `restore()` is
  idempotent; main call sites wrapped in `try/finally`.  More
  robust under `KeyboardInterrupt` and exception unwinding.
* **`_merit_jac_auto` uses `scheme='forward'` + cached `f0`** --
  scipy already evaluates `merit_fn(x)` before calling jac on the
  same `x`; that value is now passed as `f0`.  FD eval count
  drops from `2N` to `N + 2` per gradient (~30% saving for N=10).
* **Cancellation protocol** -- new `CancellableProgress` class
  (exposed at `lumenairy.CancellableProgress`) wired into all
  4 scipy callbacks in `design_optimize`.  `cancel()` causes
  scipy to stop gracefully; the post-loop final eval +
  `DesignResult` return still executes so the caller gets
  best-so-far state instead of a partial-data
  `KeyboardInterrupt`.
* **Merit propagator inconsistency warning** -- when
  `wave_propagator != 'real_lens'` AND any of
  `MultiWavelengthMerit` / `MultiFieldMerit` / `ToleranceAware
  Merit` is in use, `design_optimize` emits a `UserWarning` at
  entry pointing out that off-nominal Merit legs always use
  `apply_real_lens`, regardless of the selected propagator.
* **Shared `_build_asm_H_square` helper** -- the v4.13.0 Shack-
  Hartmann FFT batching introduced a local `_build_asm_H_for
  _lenslet` in `analysis/detector.py` that duplicated propagator
  H-build logic.  Now consolidated into `propagators/propagation
  .py:_build_asm_H_square(N, dx, z, wavelength, dtype=None,
  bandlimit=True)`, imported by `detector.py`.  Pinned at 1e-14
  against a hand-built reference.
* **`_fd_grad_pure` `validate_f0` parameter** -- opt-in
  validation gate for stale-`f0` detection on the forward path.
* **BSDF TIS shape assertion** -- `np.broadcast_to` mask
  replaced with a `ValueError` listing expected vs actual shape.

### Tier-3 cleanups

* `memory.set_max_ram` validates non-negative input (was silently
  accepting `-5` as -5 GB).  `get_max_ram` added to `__all__`.
* `MultiPrescriptionParameterization` raises `ValueError` on
  duplicate `(prescription_index, *path)` entries in `free_vars`.
* `_is_jax_prng_key` recognises JAX 0.4.20+ opaque PRNG keys via
  the canonical `jax.dtypes.issubdtype` check.
* Dtype-aware zero in `apply_mirror` and `apply_aperture`
  replaces `0.0+0.0j` literal that could upcast on JAX x32.
* `apply_mirror` aperture docstring corrected (said "ellipse";
  code computes a circle).
* Stale pyc cleanup from the v4.13 γ.1 revert.

### New performance wins (beyond audit scope)

| Hot path | Speedup |
|---|---|
| `vectorial_hfpi.accumulate_vector_to_grid` (1M paths, 256²) | **1.65 - 1.75x** (bit-exact) |
| `analysis/detector.py:shack_hartmann` scatter-back (K=4096) | **9.5 - 25x** (bit-exact) |
| `gbd.reconstruct_field_from_beamlets` (1024 beamlets, 96²) | **1.2 - 1.5x** typical, **2.3x** cache-warm (rel_err ~4e-16) |

Plus a side-effect win from the `ThinGratingDock` P1-D fix: the
dock now does one `thin_grating_efficiency_1d` call per
wavelength (full per-order matrix) instead of N sweeps through
the single-order helper.

**Deferred perf candidates** catalogued for v4.14+:
HF callable per-pixel loop (5-20x potential, needs API contract
change); fused `(sag, dz_dh)` helper for Newton iters (1.5-2x on
aspherics, needs `lenses.py:surface_sag_general` reorganisation);
analytic pure-conic intersection extending the v4.12.1 Newton-skip
(5-10x on pure-conic surfaces, needs broader root-selection
validation).

### v4.13.0 CHANGELOG hygiene

11 doc-vs-code mismatches in the v4.13.0 CHANGELOG entry have been
retroactively corrected.  The "Breaking changes" subsection now
explicitly enumerates the two breaking changes from v4.13.0
(`rcwa.py` rename without shim; `wavelength` sentinel).

## What's new in 4.13.0

**Three-phase bundle since v4.12.2: audit known-limitations closure
+ `except Exception:` sweep + Tier-2 perf wins.**  All 573 unit
tests pass; 34/34 validation files pass.

### Phase 1 — audit known-limitations closure

Closes S1, S2, S3 + L2, L3, L4, L6, L8 from the v4.12.1 pre-PyPI
audit:

* **S1: `io.storage` complex-dtype preservation** -- `save_jones
  _field_h5`, `append_plane_h5`, `_zarr_append_plane`, and the
  unified `append_plane` dispatcher gained a `preserve_dtype`
  parameter that threads through the write path.  Default
  behaviour preserved (v4.12.x promotion).
* **S2: `io.codegen` aperture-stop emission + wavelength sentinel**
  -- `_decompose_prescription` now emits an `aperture` step
  whenever `is_stop=True`; missing `wavelength_nm` raises
  `ValueError` instead of silently defaulting to 1.31 µm.
* **S3: `analysis.ghost` R/r convention** -- uppercase Fresnel
  reflectance disambiguated from lowercase curvature radius;
  locals renamed, public dict keys preserved.
* **L2: JAX complex-dtype routing** -- auto-enables `jax_enable
  _x64` with a one-shot RuntimeWarning when complex128 is
  required; `_PROPAGATE_SYSTEM_JAX_CACHE` key now includes dtype
  to prevent float32 → float64 cache aliasing.
* **L3: `PropagationResult.dy` + `Source.dy`** -- anamorphic-grid
  metadata round-trips end-to-end through the dispatch path.
* **L4: sibling mirror-guards** -- `apply_real_lens_traced_jax`,
  `apply_real_lens_maslov_jax`, and `lenses_maslov.apply_real_lens
  _maslov` all reject `is_mirror=True` surfaces consistently.
* **L6: `apply_mirror` xp + dy** -- backend dispatch via
  `_xp_of(E_in)` and a `dy` parameter for anamorphic grids;
  NumPy numba fast-path preserved.  `dy=None` reproduces v4.12.x
  behaviour exactly.
* **L8: zarr thread-safety** -- module-level `threading.Lock`
  guards the `Path.mkdir` monkey-patch in `_open_zarr_group_safe`.

### Phase 2 — `except Exception:` sweep

Non-UI library files: 99 → 3 justified KEEP-AS-IS sites.

* **~85 narrowed** to typed exception tuples covering the
  documented failure modes (pyFFTW failures, cache-clear guards,
  `system_abcd` fallbacks, glass-loader failures, etc.).
* **3 WARN-BEFORE-PASS upgrades** surfacing previously-silent
  degradations: `analysis/field.py petzval_radius` missing-glass,
  `propagators/hf.py` LG-decomposition fallback, and
  `design_optimize plane_logger` callback failures.
* **Regression budget pin:** non-UI `except Exception:` count
  must stay ≤ 15 (`tests/unit/test_audit_fixes_v4_13_0_except
  _sweep.py`).

### Phase 3 — Tier-2 performance wins

Representative speedups (perf_counter on Win11 / Python 3.14):

* **`bsdf.total_integrated_scatter`: 188x** -- per-(θ_i, θ_s)
  loop replaced with vectorised meshgrid + `np.trapz`.
* **Shack-Hartmann WFS (`analysis/detector.py`): 10.8x** --
  per-lenslet FFT loops batched as a single `np.fft.fft2` on a
  stacked `(n_lenslets², sa_pixels, sa_pixels)` array.
* **`thin_grating_efficiency_1d`: 10.5x** -- per-diffraction-order
  Python loop vectorised.
* **`seidel_field_sweep`: 4 - 72x** -- analytical chief-ray
  linearity (S1, S4 ∝ 1; S2, y_chief ∝ σ; S3 ∝ σ²; S5 ∝ σ³)
  replaces the per-field re-trace.  All field-independent work
  (glass indices, ABCD, marginal-ray trace) hoisted out of the
  loop.  Machine-precision agreement with the pre-hoist reference.
* **Freeform Chebyshev: 4.43x** -- `arccos(rho)` hoisted out of
  the per-order loop; `T_i(rho)` factors cached per polynomial
  order.
* **Coating stack (50 layers): 3.26x** -- tournament matmul
  reduction on a `(N, 2, 2)` per-layer tensor; scalar Snell chain
  uses `math.sin`/`math.asin` (bit-identical via libm).
* **`_fd_grad_pure` forward-FD opt-in: 2.23x** -- now accepts
  `scheme='central'|'forward'`, default `'central'`.  Default
  preserves bit-identical pre-v4.13 gradient values; forward
  path trades O(h²) for O(h) for an N+1 eval count (or N when
  `f0=<value>` is supplied).
* **`wave_opd_2d`: 1.89x** -- two-pass `np.unwrap(axis=...)`
  replaces the per-pixel Python unwrap loop.

### `rcwa.py` → `thin_grating.py` rename (no back-compat shim)

The file name "rcwa" was misleading: the function inside is the
**analytical scalar thin-phase grating** formula, not rigorous
coupled-wave analysis.  v4.4.0 renamed the public symbols
(`thin_grating_efficiency_1d`, `grating_efficiency_vs_wavelength`);
v4.13.0 finishes the rename at the file level:

* `lumenairy/elements/rcwa.py` → `lumenairy/elements/thin_grating.py`
* `lumenairy/ui/rcwa_dock.py` → `lumenairy/ui/thin_grating_dock.py`
  (class `RCWADock` → `ThinGratingDock`)

`import lumenairy.elements.rcwa` raises `ModuleNotFoundError`.

### Stage gates worth knowing

* **γ.1 (HFPI bincount swap) reverted before ship.**  NumPy ≥ 1.25
  vectorised `np.add.at` for complex via
  `_PyArray_UFuncBufferedAtVectorized`, making the bincount path
  a perf wash (0.4 - 1.0x) on the shipping toolchain.  No
  behavioural change vs v4.12.2 in `accumulate_to_grid`.
* **β.3 (`_fd_grad_for`) parameterised, not switched.**  Default
  stays `scheme='central'`; `design_optimize._merit_jac_auto`
  keeps the default explicitly so existing optimisation runs see
  no behavioural change.  Forward-FD is opt-in per call.

## What's new in 4.12.2

**Closes the round-5 / v4.12.1 pre-PyPI audit blockers.**  Three
documentation-drift items become true (NumPy `through_focus_scan`
H-hoist actually implemented, `through_focus_scan_jax` JIT cache
actually implemented, 878× benchmark headline reconciled to
~300×).  Cache hygiene infrastructure landed: `clear_asm_caches()`
extended, unbounded jit caches converted to LRU(32), 4 new public
`clear_*_cache()` helpers, all wired into `lumenairy_context
(clear_caches_on_exit=True)`.  All 482 unit tests pass.

### Pre-PyPI release blockers fixed

* **`pytest-benchmark` added to `dev` extra** -- the release-notes
  claim since v4.12.0 is now true; `pip install lumenairy[dev]`
  installs it.
* **`bench` pytest marker registered** so `pytest benchmarks/
  --benchmark-only` collects under `--strict-markers`.
* **Cache-clear / FFT-toggle helpers exposed at the top level** --
  `la.set_fft_auto_promote`, `la.clear_zernike_basis_cache`,
  `la.clear_lg_polynomial_cache`, and four new `la.clear_*_cache()`
  helpers (trace_jax, through_focus_scan_jax,
  propagate_through_system_jax, phase-retrieval).

### Documentation drift made true

* **NumPy `through_focus_scan` H-hoist actually implemented**.
  Input FFT, kx/ky, propagating mask, and target dtype now hoisted
  outside the z-loop.  v4.12.0's 4.7× speedup came from underlying
  pyFFTW MEASURE; the H-hoist now stacks on top.
* **`through_focus_scan_jax` JIT cache actually implemented**.
  Module-scope OrderedDict + LRU(32) caches the compiled vmapped
  kernel per signature.  First call ~150 ms; warm ~2 ms.  ~77× at
  N=64.

### Cache hygiene infrastructure

* `clear_asm_caches()` extended to also drop `_PYFFTW_PLAN_CACHE`
  + `_PYFFTW_BAD_SHAPES`.
* `_PROPAGATE_SYSTEM_JAX_CACHE`, `_TRACE_JAX_CACHE`, and the three
  phase-retrieval kernel caches converted from unbounded
  `Dict[Any, Any]` to `OrderedDict` + LRU(32).  No more compiled-
  XLA-binary leaks under iterative optimisation.
* `lumenairy_context(clear_caches_on_exit=True)` now calls all six
  `clear_*` helpers, each guarded.

### 878× → ~300× benchmark reconciliation

v4.12.1 release notes claimed `trace_jax` warm-call 878× at
127 ms → 0.40 ms.  Those don't reconcile (`127/0.40 = 317×`).
Fresh measurement: first **140 ms**, warm **0.47 ms**, speedup
**~300×**.  Corrected everywhere.

### Coverage test strengthened

Zemax coord-break STOP-marker test now actually places the
coord-break BEFORE the stop (exercises the pre-v4.11.2 off-by-one
bug); the v4.12.1 version placed it after, which didn't.

### Known limitations deferred to v4.13/v4.14

CHANGELOG documents the silent-data-loss class (S1: `io/storage`
append-side complex128 hardcode; S2: codegen aperture-stop drop +
1.31 µm wavelength default; S3: `ghost.py` R convention conflict)
and structural/latent items (L2: JAX complex64 hard-casts; L3:
PropagationResult missing dy; L4: sibling-gap remnants; L5: 346
`except Exception:` clauses; L6: `apply_mirror` array_namespace;
L8: zarr thread-safety).

## What's new in 4.12.1

**Closes the three perf items deferred from v4.12.0 plus the 14
missing regression tests from the round-4 audit.**  All 453 unit
tests pass; full validation suite (34 files / 314 tests) passes.

### Performance recovered

* **`trace_jax` warm-call: ~300x** (140 ms cold -> 0.47 ms
  warm, median of 20 warm calls on a 1001-ray AC254-100-C
  doublet) via a pytree-registered `JaxPrescription` wrapper +
  tracer-detection bypass that preserves `jax.grad` semantics.
  v4.12.0 reverted this because the flat-tuple cache produced
  `jax.grad = NaN`; root cause turned out to be a JAX lstsq
  backward bug, not the cache key.  Fix: bypass the jit-cache
  layer when any leaf is a tracer.  (The pre-release notes
  quoted 878x with inconsistent absolute timings; v4.12.2
  reconciled this after re-running on a stable system state.)
* **Raytrace Newton spherical fast-path: 1.50x** on 1k-ray
  doublet traces.  v4.12.0 attempted 1.64x by skipping Newton
  AND switching to the analytic spherical normal; the normal
  change compounded a 1.17e-3 NumPy<->JAX drift through Maslov
  asymptotic.  v4.12.1 ships only the Newton skip; the normal
  computation stays bit-identical to v4.11.2.
* **B1-10 half-pixel grid drift unified** -- five propagator-
  family files (gbd, mhs x2, subaperture, optimize/core) switched
  from cell-centred to pixel-centred convention, matching the
  library-wide ASM / Fresnel / RS / sources standard.

### 14 round-4 coverage-gap tests added

Each v4.11.2 fix that landed in code without a pin now has one:
`compute_psf` non-square pupil, `apply_detector` non-integer
ratio, `find_best_focus` NaN guard, MC tolerancing `a_k>=0`
clamp, `load_material` dispersion-drop warning, `Source.*`
`**factory_kwargs`, `apply_real_lens_traced` M_x/M_y, NaN
sentinel mask, `stop_index` warns, freeform RuntimeWarning,
Zemax coord-break STOP, JAX<->NumPy phase-retrieval parity,
Cassegrain S1/S2/S3/S5 hand-derived, Richards-Wolf vs paraxial
Airy at low NA.

### Other

* 2 weak v4.11.2 tests strengthened (behavioural pins replace
  source-string scans).
* Residual `try/except: skipped` removed from
  `validation/io/test_io.py:196`.
* Both cross-backend critical tests that caught v4.12.0's reverts
  now pass: `aberration_tensor_lg00_jax matches NumPy`
  rel_err = 4.53e-04; `jax.grad through fit_canonical_polynomials
  _jax is finite` grad = 1.0274e+04.

## What's new in 4.12.0

**Combined performance pass + round-4 pre-PyPI audit.**

### Performance — Tier-1 speedups (vs v4.11.2)

* **ASM propagate 1024^2: 4.3x** -- pyFFTW double-buffer cache +
  auto-promote ESTIMATE -> MEASURE.
* **`through_focus_scan` 7-pt N=256: 4.7x** -- input FFT and
  kx/ky hoisted outside the z-loop.
* **`propagate_through_system_jax` warm: 163x** -- module-scope
  jit caches at the JAX system entry.
* **`gerchberg_saxton_jax` / `error_reduction_jax`: 36-46x**
  steady-state -- jit-cached iteration kernel.
* **`propagate_hf_chebyshev_quadrature`: up to 22x** -- replaced
  scalar pixel loop with vectorised chunk broadcast +
  `np.einsum`.
* **`zernike_basis_matrix` warm cache: ~12000x** (22 ms -> 1.8 us);
  `zernike_decompose` 10-call loop: 3.7x.

The 4.7x `through_focus_scan` speedup propagates through
`MultiWavelengthMerit` / `MultiFieldMerit` / Monte-Carlo
tolerancing: **100-trial MC at 31-pt N=256 drops from 71 s to
15 s**.

### Round-4 audit: ~20 PyPI release blockers fixed

* **README cookbook examples now runnable** -- 11 broken
  positional / renamed / missing-kwarg calls fixed.
* **Back-compat deprecation shims** for `load_zmx_prescription`
  and `load_zemax_prescription_txt`.
* **JAX `propagate_through_system_jax`** unified aperture
  schema with NumPy; raises `NotImplementedError` up-front for
  non-traceable element types.
* **Rayleigh-Sommerfeld `z<=0` guard** matches Fresnel /
  Fraunhofer / SAS.
* **Dispatcher negative-z routing** and **`output_grid` for
  ASM family** correctly auto-promote / raise.
* **`_apply_doe_kick_jax` gradient flow** preserved.
* **`makedammann2d` global RNG** no longer mutated.
* **`image_plane_wfe` reference-sphere `1/N_chief`** for off-
  axis fields.
* **`distortion_grid` `L^2+M^2 < 1` guard**.
* **`apply_real_lens_traced` mirror guard** with a properly
  named error.
* **`gerchberg_saxton(backend='jax')`** forwards
  `seed`/`dtype`/`initial_phase` end-to-end.
* **`ghost.py` docstring** clarifies the `'intensity'`
  upper-bound caveat.

### Deferred to v4.12.1

* Raytrace Newton spherical fast-path (caused a 1.17e-3
  NumPy<->JAX drift).
* `trace_jax` jit cache (broke `jax.grad(fit_canonical
  _polynomials_jax)`).
* B1-10 half-pixel grid convention drift between propagator
  families.

### Tooling

* New `benchmarks/` directory with `pytest-benchmark` per-area
  perf tests.
* All 390 unit tests pass; full validation suite (34 files /
  314 tests) passes.

See `CHANGELOG.md` for the full per-finding breakdown.

## What's new in 4.11.2

**Round-3 fresh-eyes audit response.**  An 11-agent fresh-eyes audit
of v4.11.1 surfaced ~120 new substantive findings.  v4.11.2 closes
~70 of the highest-impact findings across seven parallel work
tracks plus reconciliation, and ships ~55 new pinning regression
tests.

Headline meta-finding: **the v4.10 "C-LR-1 fix" was itself wrong.**
The pre-v4.10 sign on the Seidel-correction OPL inside
`apply_real_lens` was correct; the round-1 audit's physics-reasoning
step that justified flipping it was reversed.  v4.11.2 restores the
original sign and pins it with a ground-truth regression test
against `apply_real_lens_traced`.

Other highest-impact fixes:

* **GBD axial-OPL fix activated** (v4.11.1 version was calling `.get`
  on a dataclass — silently swallowed AttributeError, axial_opl
  always None).
* **S-LAH64 / S-LAH79 Sellmeier coefficients removed** (in-code values
  were off by 5-6% in n_d; now route through refractiveindex.info).
* **Chained-mirror Seidel parity** tracked across `system_abcd` and
  `seidel_coefficients`; Cassegrain / Schwarzschild correct now.
* **EVENASPH PARM off-by-one** in Zemax loader fixed (every Zemax-
  authored EVENASPH file previously lost its α₄ on load).
* **Quadoa aspheric serializer** wrote powers instead of values.
* **`normalize_prescription` mirror filter** was a no-op (wrong key).
* **HFPI finite-conjugate path** was killing all rays; **stratified
  sampler** was sampling only 2 of 16 strata.
* **Richards-Wolf prefactor** `1/f²` + sign restored.
* **`compute_psf` Parseval test** was passing for the wrong reason.

~55 new regression tests; full validation suite (34 files, 314 tests)
passes.

See `CHANGELOG.md` and `AUDIT_ROUND3_2026_05_16.md` for the full
per-finding breakdown.

## What's new in 4.11.1

**Round-2 verification follow-up.**  An independent verification of
the v4.10 / v4.11.0 fix wave identified five fixes that had landed
dead-on-arrival (call-signature mistakes / wrong-API lookups),
several new bugs the fix wave introduced, three unfixed JAX silent-
failure paths, and one over-coarse threshold.  4.11.1 closes all of
these and ships the first round of pinning regression tests.

* **MultiWavelengthMerit** chromatic optimisation actually runs now
  (was a no-op for the entire v4.10 series -- positional call to a
  keyword-only `apply_real_lens` raised TypeError on every iteration,
  swallowed by a bare except).
* **Decentered aperture stop** is honoured (the 4.10.2 fix called
  `getattr` on a dict with the wrong key names and was inoperative).
* **Circular-polarisation handedness** consistent across
  `create_circular_polarized`, `apply_waveplate`, `stokes_parameters`,
  and `vector_diffraction.py` -- all three sites now agree on
  `S3 > 0 = "right"`.
* **Richards-Wolf rim mask** built before clipping (was identically
  True over the whole grid -- geometric pupil unenforced).
* **JAX intersect alive-masking** propagates disc<0, non-finite t,
  and Newton-stuck rays into `state.alive`.
* **`_refract` / `_reflect`** clear `rays.alive` (not only
  `error_code`) on direction-vector collapse.
* **`create_point_source`** central pixel is now bounded by
  `amplitude / dx` (was 1e30 from a `1e-30` floor on r).
* **GBD-through-prescription** populates `axial_opl`; reconstruction
  carries the system's absolute axial phase reference.
* **Test coverage**: 9 new pinning regression tests
  (`tests/unit/test_audit_fixes_v4_11_1.py`).  All 179 unit tests
  pass; the v4.10 series shipped with zero new test files.

See `CHANGELOG.md` for the full per-finding breakdown.

## What's new in 4.11.0

**Multi-agent physics audit response — full series.**  4.11.0 closes
the converged findings from three independent audit runs (one
external 8-agent audit, two internal multi-agent audits) of the
v4.9.0 codebase.  Across five patch releases (4.10.0 → 4.10.1 →
4.10.2 → 4.10.3 → 4.11.0) **~100 audit findings** were addressed.
All 34 validation files (314 tests) pass.

### Highest-impact fixes

* **Mirror Seidel coefficients no longer zero.**  Every reflective /
  catadioptric system silently reported "diffraction-limited"
  Seidel sums pre-4.11 because the mirror branch in
  ``seidel_coefficients`` updated ray heights but never wrote
  S1..S5.  Fixed via Welford form with n2 = -n1.
* **Exit-pupil radius** uses transverse magnification 1/D (was D,
  the angular magnification).  Off by 1/D² for non-trivial post-
  stop systems.
* **Tilted-ASM band-limit** now centred on the original-frame
  spectrum FX + fx0.  Pre-4.11 the default ``bandlimit=True``
  zeroed the propagated field for any non-trivial tilt.
* **Rayleigh-Sommerfeld kernel sign** flipped to the Goodman 3-43
  form (1/r − ik).  Coherent superposition of RS with ASM /
  Fresnel was 180° out of phase pre-4.11.
* **Richards-Wolf** adds the missing 1/√(cos θ) Jacobian and the
  -ikf/(2π)·exp(-ikf) prefactor.  High-NA focal-plane amplitudes
  are now physical.
* **Coord-break order** matches Zemax PARM 6 default (decenter-
  then-tilt).  Imported folded designs now get the correct frame
  transform.
* **``MultiWavelengthMerit``** actually re-propagates the wave leg
  per wavelength (was a no-op chromatic constraint pre-4.11).
* **``create_circular_polarized('right')``** returns the RHC Jones
  vector (1, -i)/sqrt(2) under the library's exp(-i omega t)
  convention.
* **JAX trace correctness floor**: sign(R) on sag derivatives, TIR
  double-where for finite ``jax.grad``, ``trace_jax`` raises on
  unsupported surface types instead of treating them as flat
  refractive.
* **HFPI** Kirchhoff weighting now includes the 1/(i*lambda)
  prefactor and solid-angle Monte-Carlo weight.  Absolute
  amplitudes were unphysical by ~10^6 pre-4.11.
* **Shack-Hartmann** drops the bogus wavelength/(2π) factor on
  wavefronts; reference-centroid calibration pass added.
* **``through_focus_scan_jax``** now uses ``jax.vmap`` (was a
  Python for-loop) -- ~5-15x speedup, more on GPU.

### New / improved API

* ``ChromaticFocalShiftMerit(wavelengths=[...])`` is self-contained.
* ``check_sampling_conditions(NA=...)`` for relaxed Nyquist.
* ``register_fixed_glass`` works without ``refractiveindex`` installed.
* Phase-retrieval functions accept ``seed=`` and ``dtype=``.
* Source factories (``create_top_hat_beam``, ``create_annular_beam``,
  ``create_bessel_beam``) accept ``dy`` for anamorphic grids.
* ``apply_real_lens`` respects decentered stop apertures.

### Documented limitations (carried forward)

* ``_transfer_jax`` uses the paraxial form ``x += L·thickness``.
  The math-correct ``t = (thickness − z)/N`` introduces a gradient
  instability through ``fit_canonical_polynomials_jax`` whose root
  cause needs deeper investigation.  Paraxial form is accurate to
  ~1 % for NA ≤ 0.1.

See [CHANGELOG.md](CHANGELOG.md) for the full per-release fix list
and the explicit "not addressed" items with rationale.

## What's new in 4.9.0

**External-audit response + scoped runtime-environment manager.**
4.9 bundles the 4.8.1 ``lumenairy_context`` work with correctness
fixes from the v4.8.0 external audit
(``LumenAiry_Audit_Report.md``).  170 unit tests + 34 validation
files pass.

### Audit fixes (physics)

* **#2.1 Seidel formula** -- ``seidel_coefficients`` now uses
  ``Δ(u/n) = u_after/n2 − u_before/n1`` (Welford 8.46), not the
  pre-4.9 ``Δ(1/n)`` shorthand.  Buggy magnitudes were off by 1.5×-5×
  per surface depending on incidence angle / index; the flat-
  refracting-surface branch (which zeroed S1/S2/S3) is also fixed.
* **#2.5 ``aberration_tensor`` ℓ ≠ 0 outputs** -- pre-4.9 the
  chief-ray projection collapsed to the constant term of the LG
  output polynomial, silently returning zero for any ℓ ≠ 0 mode
  (coma, astigmatism, tilt).  4.9 does the full output-plane
  σ-integration via ``propagate_modal_asymptotic`` + ``decompose_lg``;
  ℓ ≠ 0 modes now carry real physical meaning.
* **#4.6 / #4.7 ``seidel_wfe`` Petzval H²** -- uses the Lagrange
  invariant ``|H|²`` instead of bare ``sigma²`` (off by (D/2)² ~
  100× for a typical singlet); S5 Schwarzschild relation also
  picks up the missing H² factor.
* **#2.2 GBD axial phase** -- ``exp(1j·k·t)`` instead of
  ``exp(1j·k·abs(t))``; back-propagation now round-trips.
* **#2.3 Coronagraph λ/D scaling** -- ``coronagraph_contrast_curve``
  now uses ``pupil_diameter_m`` to compute the correct
  ``pix_per_lam_over_D = λ·f/(D·dx_focal)``; pre-4.9 hard-coded
  ``N`` (correct only for FFT-natural pitch).
* **#3.3 Fresnel/Fraunhofer/SAS z<=0 guards** -- these forward-only
  propagators now raise ``ValueError`` on negative or zero z with
  a pointer to ASM / RS for back-propagation.
* **#3.5 TIR mask placement** -- runs for ``slant_correction=True``
  even when ``fresnel=False``.
* **#4.5 Cosmic-ray scaling** -- new
  ``cosmic_ray_rate_per_m2_per_s`` kwarg scales by detector area ·
  exposure time; legacy ``cosmic_ray_rate`` deprecation-warns.

### Audit fixes (small / documentation)

* **#4.3** ``dx > 1 mm`` validator loosened to ``dx > 100 mm``
  (was rejecting large-telescope pupils).
* **#4.4** Zemax INCH / INCHES aliases added.
* **#2.4 / #3.1 / #3.2 / #3.4 / #4.2** docstring warnings on
  coating Snell simplification, HFPI absolute-amplitude
  non-normalisation, GBD position-only limitation, real-lens
  scalar Fresnel applicability, and LG polynomial normalisation.

### New (4.8.1 work bundled in)

* **``lumenairy_context(**kwargs)``** -- scope a block of code
  with isolated library runtime settings (``complex_dtype``,
  ``pyfftw_planner``, ``fft_threads``, ``max_ram``, ASM cache
  caps).  Nests cleanly, restores on exception, optional
  ``clear_caches_on_exit=True``.
* **``dtype=`` kwarg on the 11 source factories** --
  ``create_gaussian_beam(..., dtype=np.complex64)`` allocates
  explicitly; default ``dtype=None`` inherits from the global
  default.
* **``atexit`` auto-restore** -- import-time defaults snapshot
  on first ``import lumenairy``; restored on process shutdown to
  catch the long-Jupyter-session foot-gun where
  ``set_default_complex_dtype`` would otherwise leak.
* **New getters**: ``get_pyfftw_planner``, ``get_fft_threads``,
  ``get_asm_cache_size``, ``get_max_ram``.

---

## What's new in 4.8.0

**Library-correctness pass triggered by the external wiki audit.**
4.8 fixes three verified bugs surfaced by the audit and closes the
two long-deferred items (`DeformableMirror._IF_basis` memory
foot-gun from the 4.0.1 deferred list; folded-design wave-optics
silent-drop from the 3.7.8 archive page).  All fixes ship with
regression tests; 119 unit tests + 34 validation files pass.

### Bug fixes

* **`create_point_source` sign convention.**  The formula
  `E = A * exp(+i*k*r) / r` was sign-independent in `z0` and always
  produced a diverging wave even when the caller requested
  converging.  Fixed: switch the exponent sign based on `z0` —
  `z0 <= 0` → `exp(+i*k*r)/r` (diverging, source before grid),
  `z0 > 0` → `exp(-i*k*r)/r` (converging, focus after grid).
  **Behaviour change:** code that used `z0 > 0` to model an
  *outgoing* wave needs to flip to `z0 < 0`.
* **`FocalLengthMerit` / `BackFocalLengthMerit` afocal target.**
  Docstring promised "pushing EFL toward infinity drives merit to
  zero" but the code returned `weight * efl**2`, growing without
  bound.  Switched the `target == 0` branch to penalise optical
  power `(1/efl)^2`, so merit → 0 as `efl → ∞`.  Optimisers
  targeting a collimator now have the correct gradient.

### Long-deferred items now fixed

* **`DeformableMirror._IF_basis` memory foot-gun.**  Eager
  pre-allocation of the `(n_act, n_act, N, N)` influence-function
  stack was 8 GB float64 for a 32×32 actuators × 1024×1024 pupil
  config (deferred since the 4.0.1 audit).  New
  `cache_basis = {'auto', True, False}` flag (default `'auto'`)
  caches eagerly below a 512 MB ceiling and switches to on-demand
  evaluation past that — same FLOP count, no large allocation.
  New `DeformableMirror.fit_phase(target_phase)` public modal-to-
  zonal projection helper, replacing the worked-example pattern
  of reaching into the private `_IF_basis` attribute.
* **Wave optics through folded designs silent-drop.**
  `apply_real_lens`, `apply_real_lens_traced`, and
  `apply_real_lens_maslov` walked `prescription['surfaces']`
  (refracting-only) and silently ignored mirrors in
  `prescription['elements']`; for a folded `.zmx` this propagated
  the unfolded-equivalent path with the mirror's focusing phase
  and world-frame axis change dropped.  All three entry points
  now raise a `ValueError` with a clear message unless the caller
  acknowledges the unfolded-equivalent treatment via
  `prescription['allow_unfolded_equivalent'] = True`.  New public
  helpers **`split_prescription_at_mirrors(rx)`** and
  **`has_mirrors(rx)`** support the explicit segment-by-segment
  workflow.

### Wiki audit closeout (cross-cutting)

The companion wiki underwent a comprehensive review pass: all 20
critical, all 58 major, and ~290 of 294 medium findings addressed
across 18 commits.  Highlights: ASM aliasing thresholds documented
in Tutorial / Quickstart; test-count contradictions reconciled
(34 files / ~700 tests / 119 unit); Physics XII §1 / §5 photon-noise
formulas corrected; Marechal-Strehl exponent fixed; Fresnel-number
aliasing criterion corrected `N²/4 → N/4`; new FR-Memory page
documenting RAM + FFT-planner helpers.  The full audit response is
documented in the wiki [Release Notes](../../wiki/Release-Notes#whats-new-in-4-8-0).

---

## What's new in 4.7.0

**Polish-pass release + API-consistency pass.**  4.7 implements the
verified items from the polish-pass audit, including the J.2
API-consistency work scheduled for this milestone.  **Breaking
changes** (no deprecation cycle, since the library has a single
user at present): see below.

### Breaking changes

* **Lens-function args are keyword-only past `E_in`** -- all 8
  `apply_*_lens` entry points (`apply_thin_lens`, `apply_spherical_lens`,
  `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply_grin_lens`,
  `apply_real_lens`, `apply_real_lens_traced`, `apply_real_lens_maslov`)
  now enforce keyword-only arguments after the input field.  This
  removes the positional inconsistency where `wavelength` lived at
  pos 3 in some, 5 in others, and 6 in `apply_spherical_lens`.
* **`apply_real_lens` trio: `lens_prescription=` → `prescription=`**
  (matching the rest of the library, 54 prior uses).
* **Diffractive-lens factories drop the `_m` suffix** --
  `create_diffractive_lens`, `create_kinoform`,
  `create_fresnel_zone_plate` now take `dx`, `focal_length`,
  `wavelength` (no suffix).  Library uses SI metres throughout.
* **Source-factory ordering standardised** on `(N, dx, wavelength, *,
  source_specific)`.  `sigma` / `diameter` / `outer_diameter` /
  `inner_diameter` / `mode_field_diameter` are now keyword-only.
* **`wavelength` is now required** (no default) on
  `keplerian_telescope`, `beam_expander_prescription`, every Zemax /
  CODE V / Quadoa exporter, and `make_ray`.  Removes the disagreeing
  defaults (550 nm vs 1310 nm) that surfaced in the audit.
* **Zemax loaders renamed**: `load_zmx_prescription` →
  `load_zemax_zmx`; `load_zemax_prescription_txt` →
  `load_zemax_prescription_data_txt`.

### Added

* **Input validation** on every public propagator and on
  `surfaces_from_prescription` -- catches `wavelength = 0`,
  `wavelength = 1.31` (forgot units), `dx <= 0`, malformed
  prescriptions, NaN inputs, with messages that quote the
  parameter, value, and calling function.
* **Glass-registry rebuild** -- callable entries via
  `GLASS_REGISTRY[name] = lambda wl: n`; 30 new bundled Sellmeier
  glasses (N-FK51A, N-SF57, N-LASF44, S-LAH64, ...); `list_glasses()`
  / `search_glasses()` helpers; typo suggestions on unknown names.
* **`dy=None` kwarg** on the `apply_real_lens` trio for API symmetry.
* **Dataclass returns** for `distortion_grid`,
  `footprint_per_surface`, `spot_diagram_vs_field` (with
  dict-style subscript preserved).
* **Canonical-order propagator aliases**: `propagate_gbd`,
  `propagate_hfpi`, `propagate_huygens_fresnel` all share
  `(E_in, z, wavelength, dx, ...)` matching ASM.
* **`asm_propagate` + `which_propagator`** -- auto-selector and
  advisor for the ASM family (plain / tilted / MFT / SAS /
  Fraunhofer) based on geometry.
* **`plot_lens_layout(prescription, ...)`** -- script/notebook
  cross-section drawing extracted from the GUI's `Layout2DView`.
* **`plot_glass_map` + `abbe_diagram`** -- glass-catalogue
  scatter / dispersion plots extracted from the GUI's
  `glass_map_dock`.
* **`py.typed` marker** so type checkers honour the library's
  hints.  4.7's annotation pass took coverage from 28.5% to
  90.2% (>=partial) and 10.5% to 70.1% (fully annotated).
* **`pytest validation/`** -- the 670-test validation suite is
  now pytest-discoverable via a `validation/conftest.py` adapter.
  The legacy `python validation/run_all.py` driver keeps working;
  the new path adds IDE test discovery, `-k` filtering, JUnit XML,
  and parallel execution via `pytest-xdist`.
* **`backend='jax'` kwarg** unifies the `_jax`-suffixed analysis
  siblings (Gerchberg-Saxton, error reduction, HIO,
  through-focus scan, Monte-Carlo tolerancing).
* **Array-namespace dispatch documented**: passing a CuPy or
  JAX array as `E_in` automatically routes the entire pipeline
  through that backend.  `use_gpu=True` remains as a back-compat
  way to force NumPy→GPU promotion.

### Packaging hygiene

* Module-level `__all__` in every analysis submodule.
* `lumenairy.analysis.analysis` → `lumenairy.analysis.core` (with
  back-compat shim).
* PyPI `Changelog` / `Releases` URLs in `pyproject.toml`.

* **Propagator input validation.** Every public propagator (ASM,
  Fresnel, Fraunhofer, RS, scalable ASM, the MFT variants, batch and
  tilted) now calls a shared `_validate_propagator_inputs` helper at
  the entry point.  It catches the silent-failure regimes the old
  code shipped with -- `wavelength = 0` (was `ZeroDivisionError`),
  `wavelength = 1.31` (forgot the e-6, was silent garbage),
  `dx = 0` (was `ZeroDivisionError`), `dx = 2.0` (forgot units, was
  silent garbage), 3-D / 1-D / empty inputs, non-finite `z`.  The
  error message quotes the parameter, the value, and the calling
  function.
* **Prescription validation.** New public
  `lumenairy.validate_prescription(prescription, *, strict=True)`,
  also called internally by `surfaces_from_prescription`.  Catches
  empty dict, missing keys, surface/thickness length mismatch, NaN
  radius, bad aperture, etc., with a precise message.
* **Glass-registry rebuild.** `GLASS_REGISTRY` entries can now be a
  custom *callable* `f(wavelength_m) -> n_real_or_complex` so you
  register custom dispersion (Cauchy, temperature-dependent, etc.)
  with a one-line lambda.  A bundled `SELLMEIER_COEFFICIENTS` table
  adds ~30 Schott / Ohara entries (N-FK51A, N-SF57, N-LASF44, S-LAH64,
  etc.) usable without the `refractiveindex` package.  `list_glasses()`
  / `search_glasses(pattern)` helpers + typo suggestions on unknown
  glass names.
* **`dy` kwarg on the lens trio.**  `apply_real_lens`,
  `apply_real_lens_traced`, and `apply_real_lens_maslov` now take
  `dy=None`.  `apply_real_lens` honours non-square pixels through
  the per-surface phase screens and in-glass ASM; the traced / Maslov
  variants accept the kwarg and raise on `dy != dx`.
* **Field-analysis dataclass returns.** `distortion_grid`,
  `footprint_per_surface`, and `spot_diagram_vs_field` now return
  named dataclasses (`DistortionGrid`, `SurfaceFootprint` +
  `FieldFootprint`, `SpotDiagramField`) -- closing the inconsistency
  with the rest of the 4.4 field-analysis suite.  Dict-style
  indexing keeps working for back-compat.
* **Module rename.** `lumenairy.analysis.analysis` →
  `lumenairy.analysis.core`.  Back-compat shim preserves the old
  dotted import.
* **Packaging hygiene.** Module-level `__all__` in every analysis
  submodule.  PyPI `Changelog` + `Releases` URLs.

## What's new in 4.6.0

**Documentation overhaul -- decision-tree front door + lens-family
cross-refs.**  No code changes; the public API is identical to 4.5.0.

* **README rewritten around "which function should I use?"** -- the
  former Quick Start was replaced with a decision-tree-style set of
  "if you need X, use Y" tables covering free-space propagation,
  lens application (the three `apply_real_lens*` variants), folded
  designs, field analysis, and design optimisation.  Three minimal
  end-to-end examples follow; longer recipes moved into a "Cookbook"
  section near the end.
* **`apply_real_lens` / `_traced` / `_maslov` docstrings cross-link
  to each other.**  Each now opens with a `See Also` block + a
  one-line selection guide so the choice between the three
  fidelity points is visible right at the top of `help(la.apply_real_lens)`.
* **Dense physics citations moved out of the README main flow.**
  Matsushima-Shimobaba, Heintzmann-Loetgering-Wechsler kernel
  details, and the per-variant real-lens accuracy strategy now live
  in an "Appendix: Physics references" at the bottom -- still here
  for anyone who wants them, just out of the way for first-read.

## What's new in 4.5.0

**World-frame ray tracing for folded prescriptions, end-to-end.**
Closes the deferred gap from 4.4: folded `.zmx` designs can now be
loaded, world-frame-traced, and analyzed paraxially from any script,
without instantiating the GUI's `SystemModel`.  Pure-additive; no
breaking changes.

* **Mirror inference (`surfaces_from_prescription`).**  A prescription
  surface with `glass_after='MIRROR'` (the Zemax convention from
  `load_zmx_prescription`) is now auto-detected and emitted with
  `is_mirror=True`, and `glass_after` is normalised back to
  `glass_before` (since reflection does not change the surrounding
  medium).  Previously this flag was silently dropped, and a folded
  prescription needed manual fix-up.  No effect on un-folded
  prescriptions.

* **`paraxial_focus_world(world_surfaces, wavelength)`** -- returns
  the world-coordinate `(focus_origin, focus_normal)` of the paraxial
  image plane.  Traces a chief + paraxial-marginal ray through the
  world surfaces and finds their closest-approach point, so it works
  correctly on folded prescriptions where the unfolded BFL would
  carry the wrong sign for a direct world-frame walk.

* **Folded-design validation suite.**  New `validation/raytrace/
  test_folded_designs.py` builds a periscope (plano-convex singlet +
  45-deg flat fold mirror) and a 45-deg-tilted concave spherical
  mirror from prescription dicts, then verifies:
  * The fold mirror lands at the correct world coordinate
    `(0, 0, 53mm)` / `(0, 0, 100mm)` and the detector lands 50mm
    `(100mm)` post-fold in `+y`.
  * An on-axis chief ray reflects off the mirror and lands at the
    detector vertex.
  * `paraxial_focus_world` returns the correct image-plane world
    position for both designs.  For the curved fold the result
    matches the analytical tangential focal length
    `f_t = R/2 cos(45 deg)` ~ 70.7mm, the obliquity-induced
    astigmatism that a paraxial trace through a tilted spherical
    mirror produces.
  * Straight-axis singlets give a world focus identical to
    `last_surface_z + BFL`.

* **All 34 validation files pass** (33 from 4.4 + 1 new), 52 unit
  tests still green.

## What's new in 4.4.0

**Field-resolved analyses lifted from GUI + world-frame surface builder
for folded designs + Strehl alternatives + grating rename + glass-catalog
warnings.**  Mostly additive; one breaking name change.

* **World-frame surface builder (`la.world_surfaces_from_prescription`).**
  Previously, building world-frame surfaces from a folded prescription
  (e.g. a `.zmx` with COORDBRK surfaces) required going through the
  GUI's `SystemModel`.  4.4 lifts that translation into the library:

  ```python
  presc = la.load_zemax_zmx('folded_design.zmx')
  surfaces = la.world_surfaces_from_prescription(presc)
  result = la.trace_world(rays, surfaces, 1.31e-6)
  ```

  The builder walks the combined coord-break + optical-surface
  sequence in `surf_num` order, applies tilts (about local x/y/z) and
  decenters with Zemax PARM ordering, and emits `Surface` objects
  with `world_origin` (m) and `world_R` (3×3) populated.  For
  prescriptions without coord-breaks the result is bit-identical to
  the local-frame trace; for folded designs it gives correct
  geometry without touching the GUI.  `la.trace_world` is now also
  exported at top level.


* **Eight new field-resolved analyses (`lumenairy.analysis.field`).**
  Functions that previously lived only inside `ui/*_dock.py` are now
  first-class public API.  GUI docks still work -- they were
  refactored to delegate to these public functions.
  * `distortion_vs_field`, `distortion_grid` -- chief-ray
    f-tan(theta) distortion (sweep + 2-D grid).
  * `footprint_per_surface` -- per-surface ray footprints grouped
    by field angle.
  * `spot_diagram_vs_field` -- spot diagrams at multiple field
    angles on a common image plane.
  * `sensitivity_ranking` -- central-difference d(merit)/d(var)
    for an arbitrary parameter vector.
  * `relative_illumination` -- geometric vignetting vs field (new).
  * `field_aberration_sweep` -- real-ray sag/tan focus shifts and
    astigmatism vs field; companion to the paraxial
    `seidel_field_sweep` from 4.3.
  * `petzval_radius` -- paraxial Petzval surface radius helper.
  * Internal `_trace` dispatcher routes folded-design surface lists
    to `trace_world` and prescription-based surface lists to
    `trace`, so the same public functions handle both.

* **Alternative Strehl definitions.**  `strehl_ratio` (peak-ratio)
  is biased high on asymmetric PSFs where the peak shifts.  Two new
  textbook definitions:
  * `strehl_marechal(rms_waves)` -- closed-form
    `exp(-(2 pi sigma)^2)` approximation; ~0.82 at 1/14 wave.
  * `strehl_phase_integral(pupil)` -- exact small-aberration
    `|<A e^(i phi)>|^2 / <A>^2` (Born & Wolf 9.1.10), avoids
    peak-finding bias entirely.

* **`aberration_summary` warns loudly on unknown glass.**
  Previously a missing glass would return zero-filled Seidel
  coefficients with the error silently appended to a `notes`
  list -- making an unanalysable system look diffraction-limited.
  4.4 issues a `UserWarning` while preserving the zero-fill
  behaviour for back-compat.

* **Breaking name change: `rcwa_1d` removed.**  The function name
  advertised full Rigorous Coupled-Wave Analysis but the
  implementation has always been an analytical thin-grating scalar
  approximation.  Call `thin_grating_efficiency_1d` instead -- same
  signature, same output.  The 4.0.1 alias is now the canonical
  name.

* **Packaging fix: `validation/` and `tests/` ship in the sdist.**
  Added `MANIFEST.in` so a downloaded source distribution actually
  contains the validation suite.

* **GUI internals: four docks refactored.**  `distortion_dock`,
  `footprint_dock`, `spot_field_dock`, and `sensitivity_dock` now
  delegate to the new public functions instead of recomputing
  inline.  Rendering is unchanged.

* **Validation:** 1 new validation file (22 tests, all pass); 30
  pre-existing files still pass after the rename and dock
  refactors.

## What's new in 4.3.0

**Diffractive optics, off-axis Seidel, module organization, unit-test
layer.**  A four-stream release that closes audit gaps without
breaking any existing API.  All 31 validation files and 52 new
unit tests pass.

* **Diffractive lens trio.**  Three new factory functions in
  `lumenairy.elements.doe`, all exposed at top-level:
  * `create_diffractive_lens(N, dx_m, focal_length_m, wavelength_m)`
    -- continuous-phase thin-lens equivalent
    `exp(-i k r^2 / (2 f))`.
  * `create_kinoform(N, dx_m, focal_length_m, wavelength_m, n_levels=8)`
    -- quantized-phase lens.  Efficiency
    `eta ~ sinc^2(1/n_levels)`: ~81% at 4 levels, ~95% at 8, ~99%
    at 16.
  * `create_fresnel_zone_plate(..., binary=True, n_zones=None)` --
    classical amplitude FZP (`binary=True`, default; ~10%
    efficiency) or Rayleigh-Wood phase FZP (`binary=False`; ~40%
    efficiency).  `n_zones` crops to a finite aperture.

* **Off-axis Seidel analysis (`lumenairy.raytrace.seidel_analysis`).**
  The existing `seidel_coefficients` is on-axis only by design.
  Two new helpers extend it without breaking it:
  * `seidel_field_sweep(surfaces, wavelength, field_heights)` --
    evaluates the five Hopkins sums at a grid of field heights in
    one call.  Returns per-surface arrays of shape
    `(N_surfaces, N_fields)` and total sums of shape
    `(N_fields,)`, suitable for plotting S1-S5 vs field angle.
  * `seidel_wfe(seidel_result, rho, theta, ...)` -- reconstructs
    `W(rho, theta)` from a Seidel total dict using the standard
    Welford expansion (S1 rho^4 / 8 + S2 rho^3 cos / 2 + S3 rho^2
    cos^2 / 2 + S4 sigma^2 rho^2 / 4 + S5 rho cos / 2).
  * `seidel_coefficients` now returns the `field_angle` used in
    the result dict so `seidel_wfe` can apply the Petzval `sigma^2`
    scaling automatically.

* **Module organization polish.**  Two functions that lived in
  the "wrong" subpackage are now in their conceptual home;
  back-compat shims keep every existing import path working:
  * `lumenairy.ao` -> `lumenairy.analysis.ao`.  AO primitives
    (`DeformableMirror`, `apply_dm`, `LeakyIntegrator`,
    `zernike_modal_basis`, `slope_to_modal`) are an analysis /
    control layer, not a top-level element family.  Old import
    path `from lumenairy.ao import ...` still works.
  * `coronagraph_contrast_curve` moved from
    `lumenairy.elements.elements` to a new
    `lumenairy.analysis.coronagraph` (it's a post-processing
    analysis function, not an element factory).  Old import
    paths still work via a deferred-import shim.
  * New `lumenairy.elements.coronagraph` namespace module
    re-exports the four coronagraph builders
    (`apply_lyot_focal_plane_mask`, `apply_vortex_phase_mask`,
    `apply_lyot_stop`, `apply_apodized_pupil`) for discoverability.

* **Unit-test layer (`tests/unit/`).**  Five new modules
  (`test_elements_lens.py`, `test_propagation.py`, `test_sources.py`,
  `test_analysis.py`, `test_raytrace.py`) plus a shared
  `tests/conftest.py` give 52 fast API-contract tests that finish
  in <1 second total -- complementing (not replacing) the
  ~30-minute integration validation suite.  The existing
  pytest-wrapper around `validation/` moved to
  `tests/integration/test_validation_files.py`.  Run the unit
  layer alone with `pytest tests/unit`; run everything including
  the validation subprocess wrapper with
  `pytest tests/ -m "unit or integration"`.

* **Pure-additive overall.**  Every existing import path
  (`lumenairy.ao`, `lumenairy.elements.coronagraph_contrast_curve`,
  the unchanged `seidel_coefficients` shape) still works.  No
  breaking changes.

## What's new in 4.0.1

**Bug-fix patch from the post-4.0 deep audit.**  A multi-agent
correctness / performance / API / scope review of 4.0.0 surfaced
five real or latent bugs.  This release ships fixes + regression
tests for each.  Pure-additive; no breaking changes.

* `eval_image_plane_wfe` chief-ray N-fallback now uses
  `abs(N) < eps -> nan` instead of an exact-zero check with a
  non-physical unit-vector default (4.0's off-axis path had this).
* Ray-trace `_refract`/`_reflect` renormalisation no longer
  silently promotes a zero-magnitude direction-cosine to a bogus
  unit vector -- it flags the ray dead with `RAY_NAN`.
* `BSDFModel` base class is now an explicit `abc.ABC`; direct
  instantiation raises `TypeError` at construction instead of
  deferring to a `NotImplementedError` at first method call.
* `thin_grating_efficiency_1d` is a new honest-name alias for
  `rcwa_1d`, which is an analytical thin-grating scalar
  approximation (not full RCWA) -- the name `rcwa_1d` was
  misadvertising.
* `create_hermite_gauss` / `create_laguerre_gauss` docstrings now
  call out the normalisation inconsistency with the asymptotic-
  modal-propagator's analytical normalisation (a documentation
  fix, not a code change).

29/29 validation files still pass; 4 new regression tests guard
each bug.

## What's new in 4.0.0

**Polish + Tier-1-gap-closing release.**  A four-tier audit of the
library (API consistency, sign / unit conventions, cross-module
pipelines, peer-library feature parity) surfaced a set of verified
bugs and genuine functional gaps; 4.0 ships fixes for all of them
plus a batch of helpers that compose with the existing infrastructure.
Pure-additive with no breaking changes.

**Adaptive optics primitives (new module `lumenairy.ao`).**  Closed-
loop AO building blocks: `DeformableMirror` (Gaussian-IF actuator
grid), `zernike_modal_basis` + `slope_to_modal` (SH-WFS modal
reconstructor), `LeakyIntegrator` (control law).  Compose with the
existing `shack_hartmann` and `generate_turbulence_screen` into a
full single-conjugate AO loop.  See [Function Reference - Adaptive
Optics](https://github.com/travaj24/LumenAiry/wiki/Function-Reference-Adaptive-Optics).

**Off-axis image-plane WFE.**  `eval_image_plane_wfe` now accepts
`field=(Hx, Hy)` + `field_max_m` for any field point (previously
raised `NotImplementedError` for non-zero field).  New
`field_grid_wfe(prescription, wavelength, field_max_m, n_field)`
runs the WFE across an `n_field x n_field` grid and returns the
PV/RMS/Strehl maps -- the standard "image quality across the field"
plot in one call.

**More polish from the audit.**  `coronagraph_contrast_curve` for
radial contrast in lambda/D units; `normalize_prescription` to
unify schemas from the various loaders; paraxial helpers
(`field_of_view`, `f_number`, `optical_invariant`,
`defocus_waves_to_zernike`); `JonesField.stokes_parameters()` +
`.degree_of_polarization()` + `.polarization_ellipse()` bound
methods; `apply_zernike_aberration` and the HG/LG mode generators
gain `dy` kwargs for rectangular grids; storage now supports
`preserve_dtype=True`; detector model adds hot pixels, cosmic rays,
and Bayer-pattern colour filters.

**Backend preservation through analysis.**  `beam_centroid`,
`beam_d4sigma`, `beam_power`, `strehl_ratio`, `compute_psf`,
`compute_otf`, `compute_mtf`, `apply_aperture`, and
`apply_gaussian_aperture` now dispatch through
`array_namespace(...)`.  CuPy / JAX field inputs no longer get
silently coerced to NumPy.

**Source-normalisation control.**  `create_gaussian_beam`,
`create_hermite_gauss`, and `create_laguerre_gauss` all accept a
`normalize` kwarg with `'peak'` / `'power'` / `'none'` options.
Each function preserves its historical default for backward
compatibility; pass `normalize='power'` to homogenise across the
source family.

**Validation.** 29/29 validation files pass (up from 27/27 in
3.9.0).  Two new files (`test_ao.py`, `test_coherence.py`) plus
~30 new tests across the existing files.

**New wiki pages.** Function Reference - Adaptive Optics, Migration
Guide (POPPy / HCIPy / prysm / Optiland / rayoptics -> LumenAiry
mapping), and Glossary (sign conventions + when-to-use-which
propagator cheat-sheet).

## What's new in 3.9.0

**Coronagraph & broadband-imaging building blocks.**  Four
coronagraph templates land as first-class helpers
(`apply_lyot_focal_plane_mask`, `apply_vortex_phase_mask`,
`apply_lyot_stop`, `apply_apodized_pupil`) -- transmission filters
that compose with the existing MFT propagator family
(`fresnel_propagate_mft`, `fraunhofer_propagate_mft`,
`angular_spectrum_propagate_mft`) into a textbook focal-plane-zoom
coronagraph pipeline.  The end-to-end validation
(`validation/elements/test_elements.py`) confirms >100x on-axis
starlight suppression for a charge-2 vortex with an 0.85*D Lyot
stop.

`polychromatic_psf` complements the existing `polychromatic_strehl`
by returning the **full integrated PSF intensity map** on a common
image plane across wavelengths -- the realistic "what does a
broadband detector record" picture, including chromatic-defocus
broadening.  Returns power-normalised, peak-normalised, or raw
weighted-sum scaling, plus per-wavelength diagnostics
(centroid wavelength, per-lambda peaks, D4-sigma).

The existing `generate_turbulence_screen` (Kolmogorov / von Kármán
PSD with outer- and inner-scale support) is now better surfaced in
the documentation; the 5/3-power structure-function and outer-scale
variance behaviours are exercised by new validation tests.  The
three MFT propagators get full Function-Reference wiki coverage.

No breaking changes; all additions are pure-additive.

## What's new in 3.8.2

Adds two optional convention controls to
`lumenairy.eval_image_plane_wfe`:

- `image_plane='paraxial' | 'best_rms' | 'best_pv'` —
  select the longitudinal image-plane focus.  Default
  `'paraxial'` matches Zemax / rayoptics / Optiland defaults;
  `'best_rms'` does a closed-form shift to minimise WFE RMS
  (what a lab tech finds by maximising spot intensity); `'best_pv'`
  numerical-minimises peak-to-valley.
- `sphere_tangent='vertex' | 'exit_pupil'` — select where on
  the chief ray the reference sphere is tangent.  Default
  `'vertex'` is the 3.8.0 / 3.8.1 convention; `'exit_pupil'`
  matches the convention rayoptics / Optiland / Zemax use
  internally.

Plus four new diagnostic fields on `ImagePlaneWFE`
(`image_plane`, `sphere_tangent`, `r_sphere_m`,
`img_d_m_paraxial`).  Validation suite extended from 11 to 18
checks; all pass.  No changes to `apply_real_lens_traced`,
`trace()`, or any other pre-3.8.2 API; defaults reproduce 3.8.0
/ 3.8.1 output bit-for-bit.

See [Release Notes / 3.8.2](https://github.com/travaj24/LumenAiry/wiki/Release-Notes#whats-new-in-3-8-2)
for full examples and the per-library cheat sheet.

3.8.1 was a one-line metadata fix (`__version__` was stale at
`"3.7.10"` in the 3.8.0 source even though `pyproject.toml`
bumped to `3.8.0`).  No API change.

## What's new in 3.8.0

Image-plane wavefront-analysis release.  Three peer-library
features that were missing from the public API up to now:

- **`lumenairy.eval_image_plane_wfe(prescription, wavelength,
  field=(0,0), n_pupil=31)`** — direct image-plane
  reference-sphere wavefront-error evaluation (the textbook
  Zemax / Code V WFE convention).  Returns an `ImagePlaneWFE`
  result with per-ray pupil coordinates, OPD in waves, alive
  mask, and `pv_waves` / `rms_waves` / `strehl` properties.
  Complements `apply_real_lens_traced` (which returns the
  lens-exit chief-relative OPL appropriate for downstream ASM /
  Fresnel propagation); the two are connected by an exact
  ray-sphere intersection internally.  Validated against
  rayoptics + Optiland + OPDPy across 13 lens forms; see
  `OPDPy_Lumenairy_Crosscheck/CROSS_CHECK_METHODOLOGY.md` in
  the companion repo.
- **`lumenairy.remove_low_order_aberrations(opd_w, px, py,
  include_r4=True)`** — best-fit subtraction of piston + tilt +
  defocus + 4th-order spherical from a scattered OPD field.
  Residual is the genuinely-higher-order aberration content
  (6th-order SA, coma, astigmatism) where ray-trace
  implementations actually diverge; this is the realistic
  apples-to-apples cross-library agreement metric.
- **`lumenairy.raytrace.first_order_data(prescription,
  wavelength)`** — single-call paraxial summary.  Returns
  `FirstOrderData` with EFL, BFL, FFL, EP/XP positions and
  radii, principal-plane offsets, working f-number, the full
  ABCD matrix, and stop index.

**GUI:** the Analysis tab's "System Summary" dock now appends
an **Image-plane WFE (on-axis)** block reporting PV / RMS /
Marechal Strehl and the after-best-fit residual, live from
the active prescription.

See `CHANGELOG.md` and `GUI_CHANGELOG.md` for full release
notes, and the 3.8.0 wiki "What's New" page for usage
examples.

## What's new in 3.7.10

Workspace reorganisation pass.  Each tab's default dock set is
restructured so the tab's headline tool gets the dominant space.
Driven by a parallel-agent survey of how Zemax OpticStudio, Code V,
OSLO, and Optiland organise their equivalent workspaces.

**GUI**

- **Wave Optics tab**: 2D layout dropped; `waveoptics` dock is
  now the dominant left ~60%; `zernike` right; field-consumer
  tools (`phase_retrieval`, `interferometry`, `jones_pupil`,
  `coherence`) tabbed below.  Addresses the explicit user
  complaint that wave-optics wasn't front-and-center.
- **Analysis tab**: 2×2 plot grid (`spot`, `rayfan`, `psfmtf`,
  `distortion`) with field-aware and aperture-coverage siblings
  tabbed in.
- **Optimize tab**: optimizer + sliders left spine; live spot /
  rayfan / PSF-MTF top-right; summary bottom-right for live
  before/after metrics.
- **Tolerancing tab**: MC histograms dominant; sensitivity
  worst-offenders right; spot as worst-trial inspector.
- **Materials tab**: catalog list + Abbe diagram as headline
  pair.
- **Design tab**: prescription editor (central) + summary
  (System Explorer analogue) + layout / layout3d tabbed
  thumbnail + library / snapshots.

The 2D / 3D layout pair is only on the **Design** tab by default
(layout3d tabbed with layout, so both share one dock slot).
Every specialty dock remains one click away via the View menu.

Existing users get a one-time **"Workspaces reorganised (3.7.10)"**
prompt offering to migrate to the new defaults; custom workspaces
are preserved either way.

See [GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full per-tab
table.

## What's new in 3.7.9

Quality-of-life pass: debounced auto-retrace, wave-optics fold
filters + saved-file loader, multi-row prescription editing,
template-driven inserts, recent-files timestamps, F1 cheatsheet.

**Library**

- `SystemModel.trace_started` signal — fires at the top of every
  `run_trace`.  Lets GUIs raise a busy cursor / status label
  immediately rather than waiting for `trace_ready` at the end.
- Saved wave-optics HDF5 / Zarr outputs now embed the full
  prescription JSON + propagator settings + lumenairy version,
  so a saved run is self-describing.

**GUI**

- **Wave Optics dock**: explicit `Unfold steering mirrors` and
  `Ignore lateral CBs` checkboxes; embedded-prescription saved-
  file loader (with one-click "Load prescription into model");
  duration forecast updated for the filtered surface count.
- **Debounced auto-retrace**: edits coalesce into one retrace
  200 ms after the last change.  Honours the existing
  `auto_retrace_mode` pref.  Status indicator + busy cursor.
- **Multi-row select in the prescription editor**: Shift+click
  / Ctrl+click rows, right-click → "Delete N Elements" /
  "Duplicate N Elements".
- **Insert → From Template**: cemented doublet, Plossl, Petzval,
  Kepler telescope (afocal, user-chosen mag), 4-f relay.  Same
  builders exposed as example-design buttons in the Welcome dock.
- **F1 / Ctrl+?**: keyboard shortcut cheatsheet from anywhere.
- **Recent files** carry timestamps now ("2h ago", "3d ago").
- **What's New modal** refreshed for 3.7.x and auto-titled from
  `__version__`.

See [CHANGELOG.md](CHANGELOG.md) and [GUI_CHANGELOG.md](GUI_CHANGELOG.md)
for the full list.

## What's new in 3.7.8

Closes the remaining 3.7.6 / 3.7.7 fold-correctness gaps and ships
three discoverability / workflow polishes.  All four ray-trace
analysis docks (Footprint, Distortion, Spot vs Field, Ray Fan) and
the Tolerance Monte Carlo now route through the world-frame trace.

**Library**

- New `ray_fan_data_world` and `opd_fan_data_world` -- drop-in
  world-frame analogues of `ray_fan_data` / `opd_fan_data`.

**GUI**

- Ray Fan dock's tangential / sagittal / OPD plots now route
  through `*_world` helpers (3.7.7 only migrated the field-
  curvature plot).
- Tolerance dock Monte Carlo properly perturbs world-frame
  geometry: each trial shifts downstream `world_origin` values
  by `delta_t * surface_i.world_R[:,2]` for every positive-
  thickness gap.  Previously, 3.7.7 documented this as
  "intentionally not migrated" because `Surface.thickness`
  perturbations are ignored by the world trace.
- 2D layout: source-preview toggle surfaced as a toolbar
  checkbox (was a buried pref since 3.6.1).
- 2D layout: new `Export…` button for SVG (vector) / PNG (2×
  raster).

**Known limitations queued for future releases**

- Wave-optics propagation through folded systems still routes
  the unfolded equivalent.  Infrastructure inventoried; ~500-
  1000 LOC core work to wire fold-aware ASM into
  `apply_real_lens_traced`.
- Non-sequential trace (beam splitters, ghosts, double-pass)
  needs an architectural overhaul; the world-frame surface
  representation gives the right foundation but is not enough
  on its own.

See [CHANGELOG.md](CHANGELOG.md) and [GUI_CHANGELOG.md](GUI_CHANGELOG.md)
for the full list.

## What's new in 3.7.7

World-frame correctness rollout to the remaining analysis docks
(footprint, distortion, spot-vs-field, rayfan field-curvature),
plus a 3D undocked-shrink rescale fix and a GUI visual-regression
test that would have caught the prior shrink regressions in 30
seconds.  Builds on the 3.7.6 `trace_world` infrastructure.

**Library — new public API**

- `SystemModel.build_trace_surfaces_world()` — public cached
  accessor for the world-frame surface list (one Surface per
  optical surface, each carrying `world_origin` + `world_R`).
- `SystemModel.build_run_trace_world_surfaces(image_distance=
  None)` — same list with an image-plane Surface appended at
  the Detector's world frame (or paraxial focus along the last
  surface's local +z).  Drop-in replacement for the manual
  `surfaces[-1].thickness = bfl` + `Surface(radius=inf, ...)`
  boilerplate every analysis dock used to write.

**GUI — analysis docks migrated**

- Footprint, Distortion, Spot vs Field, and the Ray-Fan dock's
  field-curvature plot now use `build_run_trace_world_surfaces()`
  + `trace_world()`.  Per-surface scatter plots and chief-ray
  distortion traces land at the correct world positions in
  folded designs.
- Tolerance dock and Wave Optics dock intentionally stay on the
  legacy path — see `GUI_CHANGELOG.md` for the rationale.

**GUI — 3D undocked-shrink rescale fix**

- `Layout3DView.eventFilter` adjusts the camera's `parallel_scale`
  (or `view_angle` for perspective) proportionally to the
  viewport-height ratio on every resize.  Shrinking the floating
  3D window now CLIPS the optics at its new edges instead of
  rescaling — the on-screen pixel size of every world feature
  stays fixed.

**Validation — new GUI smoke suite**

- `validation/gui/test_layout_shrink.py` runs offscreen and
  catches: layout-dock pinned-at-minimum regressions, splitter-
  won't-shrink regressions, layout-construction crashes, and
  tx71 chief-ray world-position regressions.  14 assertions;
  picked up by `run_all.py`.  Suite total now 26 files.

See [CHANGELOG.md](CHANGELOG.md) and [GUI_CHANGELOG.md](GUI_CHANGELOG.md)
for the full list.

## What's new in 3.7.6

**Sequential world-coordinate ray trace** (core) plus a folded-
design correctness pass and a long-running dock-shrink fix in the
GUI.  The core trace engine now has a second sequential trace
path (`trace_world`) that propagates rays in world coordinates
between surfaces; each `Surface` carries its absolute
`world_origin` and `world_R` (local-to-world rotation), and only
drops into local coords for the per-surface intersect / refract /
reflect step.  This eliminates the ~`1/cos(θ)` world-position
error that the legacy local-frame trace's coord-break stack
introduced for any element following a tilted mirror.

**Library changes**

- New `lumenairy.raytrace.trace_world(rays, surfaces, wavelength,
  ...)`.  Same signature as `trace()`, but expects each surface
  to have `world_origin` and `world_R` populated.
- `Surface` gains two optional fields (`world_origin`, `world_R`,
  both default `None`).  Backward compatible: existing surfaces
  / pickles / constructor calls are unaffected, and the legacy
  `trace()` path still works.
- All 25 validation files pass unchanged.

**GUI changes — folded designs**

- `SystemModel.run_trace` now uses the world-coord trace path.
  Verified on `tx4designstudy71.zmx` (a 1× telecentric design
  with a 45° fold mirror): the chief ray's world positions
  after the fold now land at each subsequent lens centre to
  floating-point precision (`SpatialFilter`, `Lens 10`,
  `Lens 11`, `Lens 12`, `Detector` all hit dead-centre).
- Layout views (`surface_frames_2d_mm` / `surface_frames_3d_mm`)
  emit one frame per actual optical surface — no separate
  cb_pre frame.  Matches the world-trace ray history one-to-one.
- Direction-coloured ray segments (forward / return / post-fold)
  in both 2D and 3D layouts (3.7.5 rolled in).

**GUI changes — dock-shrink finally fixed**

The 2D / 3D layout dock would refuse to shrink when docked, no
matter what minimum-size constraints the contained widget or
QDockWidget reported.  The real culprit was
`element_editor.setMaximumHeight(350)` on the central widget:
when the user dragged the splitter UP to shrink the layout
dock, the freed vertical space wanted to flow into the central
widget — but it was capped at 350 px, so Qt refused to move
the splitter at all.  Cap removed; the element table grows
into the freed space naturally.

- 3D layout's button row is now a `QToolBar` with built-in
  overflow popup — when the dock is narrow, hidden buttons go
  into a `">>"` menu instead of forcing the dock open.
- Shrinking now CLIPS the 2D scene (and shows scrollbars if
  needed) instead of rescaling — `Layout2DView.resizeEvent` no
  longer calls `fitInView`.  The toolbar's "Reset View" button
  still re-fits on demand.

**GUI changes — Reset to Default Layout**

- New **View ▸ Reset to Default Layout** menu action.  Captures
  the freshly-built default dock geometry on startup and lets
  the user un-float / re-tab / re-size every dock back to
  first-launch state without restarting the application.

See [CHANGELOG.md](CHANGELOG.md) and [GUI_CHANGELOG.md](GUI_CHANGELOG.md)
for the full list.

## What's new in 3.6.1

GUI update only — no core-library API changes.  Audit-driven
overhaul of the LumenAiry Designer GUI plus three hotfixes
against bugs caught after the first 3.6.1 push.

**Layout fixes**

- Sources are now drawn in the 2D and 3D layouts as per-source-
  type glyphs (bar / ellipse / disc / annulus / dot / array).
- Selecting a row in the prescription editor now highlights the
  corresponding element in both layouts.
- Source-type changes refresh the layouts immediately (no more
  200 ms debounce wait on a discrete combo change).
- Detector is now clickable from the 2D layout's image-plane
  line.
- Rays render in correct world-frame z (the previous version
  squished the entire fan into ~10 mm because it summed only
  in-glass thicknesses; the source-to-first-surface air gap
  was missed and the parallel pre-lens beam was invisible).
- 3D layout preserves orbit / zoom / pan across parameter
  edits.  Only the very first rebuild snaps to iso; the
  toolbar's "Reset View" button still works.

**Workspace UX cleanup**

- `DEFAULT_WORKSPACES` trimmed (Analysis 13→5, Wave Optics 9→4)
  with a one-time "reset to new defaults?" migration prompt for
  upgraders.
- New top-level *View > Configure Workspace Docks…* action.
- All 3.6.0 specialty docks now exposed in the View menu.

**Industry-pattern adoptions** (Optiland / Zemax / OSLO)

- **Source-preview rays** drawn from source plane to first
  surface (toggleable, off by default).
- **OSLO-style "Attach slider to this parameter…"** — right-
  click any numeric cell in the surface sub-table to promote
  it to an optimization variable and reveal the Sliders dock
  with the new slider pulsing amber for 3 seconds.

**Cleanup**

- Removed the legacy `optical-designer` console script and
  `run_optical_designer.py` launcher.  Use `lumenairy-designer`
  (or `python run_lumenairy_designer.py`) instead.  The
  QSettings storage key is unchanged so existing user
  customisations survive.

See [GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full list.

## What's new in 3.6.0

GUI update only — no core-library API changes.  Closes ~30 audit
findings raised after 3.5.9: 5 new specialty docks (Richards-Wolf
high-NA focus, Köhler partial coherence, Shack-Hartmann sensing,
LG aberration tensor, RCWA grating), Wave Optics dispatch picks up
GBD / HFPI / Huygens-Fresnel / Subaperture, Quick-run preset bar,
detector-model toggle, Insert > Source presets, F6-F10 shortcuts,
sortable shortcuts dialog, in-app What's-New modal, expanded Help
and Tools menus, optimizer redesign, run-button standardisation,
Welcome-dock + REPL improvements.

See [GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full list.

## What's new in 3.5.9

GUI update only — no core-library API changes.  Application
renamed from "Optical Designer" to **LumenAiry Designer** with
backward-compat aliases for the launcher and console-script entry
point.  GUI catch-up to 3.3-3.5 core-library work: MFT propagators
exposed in the Wave Optics dock, Bandlimit + chief-relative-OPD
checkboxes, Custom MHS chain tab, Caustic Diagnostic dock, JAX
optimizer toggle, Tools > Scale system menu, structured Tolerance
report.

See [GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full list.

## What's new in 3.5.8

Propagator-family H-cache standardisation.  The same
``_h_cache_lookup`` pattern that already speeds up
``angular_spectrum_propagate`` is now applied to the rest of the
"fast" propagator family (Rayleigh-Sommerfeld, tilted ASM, scalable
ASM, ASM-MFT), plus a Matsushima-style ``bandlimit`` kwarg on RS and
an ``'rs'`` short alias in the ``apply_real_lens`` dispatcher.

- **`rayleigh_sommerfeld_propagate`** -- adds an H-cache (FFT'd
  Green's function keyed on geometry signature; ~2-4x warm-call
  speedup at N=1024-2048), a new ``bandlimit=False`` kwarg
  (Matsushima cutoff on the padded-grid kernel; defaults to off to
  preserve the "exact Green's function" semantic), and standardised
  ``target_cdtype`` inference matching the rest of the library.
- **`angular_spectrum_propagate_tilted`** -- adds an H-cache keyed on
  ``(Ny, Nx, dy, dx, lambda, z, fx0, fy0, bandlimit, dtype)``;
  ~1.5-1.7x warm-call speedup at N >= 1024.
- **`scalable_angular_spectrum_propagate`** -- bundles the three
  per-call padded kernels (``delta_H``, ``H1``, ``H2``) under one
  cache entry; ~2x warm-call speedup across N=512-2048.
- **`angular_spectrum_propagate_mft`** -- caches the input-grid H
  separately from the per-call Bluestein step.  Calls onto multiple
  output grids from one input plane (the natural focal-plane-probing
  pattern) get **~2.7x amortised speedup** at N=1024 after the first
  build.
- **`apply_real_lens(wave_propagator='rs')`** -- short alias for
  ``'rayleigh_sommerfeld'``; the dispatcher also forwards the
  function's ``bandlimit`` kwarg into the RS path.

The internal ``_h_cache_store`` byte budget grows a small
``_entry_bytes`` helper so it correctly accounts for both single-array
entries and tuple bundles (the SAS case).  No existing API behaviour
changes; ``__all__`` unchanged.  46/46 propagator tests pass; all 25
validation files green.

## What's new in 3.5.7

Inter-library compatibility improvements.  Driven by an empirical
multi-library cross-validation (LightPipes, prysm, POPPy, diffractio,
OPDPy) that surfaced two gaps -- a phase-convention mismatch against
ray-trace-rooted aberration tools and a missing arbitrary-output-grid
focal-zoom propagator.  Both addressed; nothing existing changes.

- **`apply_fresnel_curvature(E, dx, wavelength, R, sign=±1)`** -- new
  utility for round-tripping between phase conventions.  Lumenairy and
  the Fresnel/ASM-propagator family (LightPipes, prysm, POPPy,
  diffractio, Zemax POP) keep the absolute physical phase at the
  output plane.  OPDPy and Zemax wavefront operands store the
  chief-relative OPD -- a different reference, purpose-built for
  aberration analysis -- which differs by exactly
  `exp(i*k*r²/(2R))` with `R = v - f` for a thin-lens imager
  (Gaussian-beam ABCD prediction).  The new utility converts cleanly
  in either direction.  Empirical agreement post-correction:
  Lumenairy ↔ OPDPy at **0.99996** complex correlation.

- **`fresnel_propagate_mft`** -- single-FFT paraxial Fresnel onto a
  user-specified output grid via Bluestein chirp-Z transform.  Pick
  any `dx_out`, `N_out`, `centre_out=(x, y)` independently of `dx_in`
  and `z`.  Standard tool for focal-plane zoom (sample a tightly-focused
  region at sub-pixel resolution without padding the input grid by
  the corresponding factor).  Same physics as `fresnel_propagate`.

- **`fraunhofer_propagate_mft`** -- far-field counterpart to
  `fresnel_propagate_mft`.  Excellent for coronagraph and
  high-contrast imaging workflows.  POPPy's `apply_image_plane_fftmft`
  and prysm's `focus_fixed_sampling` are well-established equivalents
  in their respective ecosystems and the inspiration for this addition.

- **`angular_spectrum_propagate_mft`** -- exact ASM
  (`exp(i*kz*z)` with `kz = sqrt(k² - kx² - ky²)`) followed by a
  Bluestein chirp-Z inverse FT onto the user-specified output grid.
  POPPy, prysm, and diffractio offer arbitrary-output-grid focal zoom
  via their paraxial Fresnel propagators (the natural choice for the
  imaging applications they're built for); this addition extends that
  capability to high-NA / strongly-diverging beams where the exact ASM
  kernel is preferred.

All three propagators share an internal `_bluestein_2d` helper module
and follow the same backend dispatch pattern as
`angular_spectrum_propagate` (NumPy / pyFFTW / CuPy / JAX, with
`jax.grad` validated to ~1e-5 of finite differences).  At the natural
FFT-output grid, MFT versions agree with their non-MFT siblings to
~2e-14 relative.  Bluestein cost scales as `O((N + M) log (N + M))`
per axis vs `O(N²M²)` for direct matrix DFT.

`__all__` grows from 391 to 395.  No existing API changes.  All 40
propagator validation tests pass.

## What's new in 3.5.6

CI hotfix + 14 audit-driven performance / accuracy improvements.

- **`trace_jax_with_params`** -- JAX-array-aware ray tracer that accepts
  radii, conics, aspheric coefficients, and thicknesses as
  differentiable JAX arrays.  Unblocks design-parameter adjoint
  optimization that 3.5.5's `make_lg_aberration_merit_jax` only
  partially supported.
- Newton iter cap reverted 8 → 12 in `apply_real_lens_traced` (3.5.5
  drop showed 0% speedup).
- Adaptive Newton convergence warning when >1% of pixels fail.
- Maslov upsample phase fix (cubic zoom of cos/sin instead of
  `np.unwrap`).
- JAX x64 auto-enable in `fit_canonical_polynomials_jax`.
- `set_pyfftw_planner('FFTW_MEASURE')` API for ~20% faster FFTs after
  ~1 s planning cost.
- Vectorised Vandermonde + 4-D polynomial evaluator (3-5× faster
  Maslov hot path).
- `design_optimize(precision='single')` for ~2× FFT throughput.
- `non_sequential_stray_light` and `monte_carlo_tolerancing_linearized`
  helpers.

Plus a CI-only fix: missing `from typing import Tuple` in
`elements/lenses.py` after the 3.5.5 split.

## What's new in 3.5.1

Patch release: additive JAX-companion functions across `analysis`,
`system`, and `propagators`.  Every change is opt-in (suffix `_jax`);
no existing function behaviour changes.

- **`solve_envelope_stationary_jax_ift`** — JAX-grad-friendly Newton
  solver for the envelope-stationary equation, with backward via the
  implicit function theorem (`jax.custom_vjp`).  Forward matches the
  NumPy solver to ~1e-12; backward IFT matches FD to single-precision
  JAX float32 floor.
- **`through_focus_scan_jax`** — JAX-batched z-scan via
  `angular_spectrum_propagate` (already JAX-traceable).
- **`gerchberg_saxton_jax`, `error_reduction_jax`, `hybrid_input_output_jax`**
  — JAX-jit'd phase-retrieval iterations using `jax.lax.fori_loop`.
- **`propagate_through_system_jax`** — element-by-element walk with
  per-element JAX dispatch for `propagate` / `lens` / `aperture` /
  `mask`; falls back to NumPy at element boundaries for unsupported
  types.

Reserved stubs documenting the planned interface (raise
`NotImplementedError`):
`fit_canonical_polynomials_jax`, `apply_real_lens_traced_jax`,
`apply_real_lens_maslov_jax`, `monte_carlo_tolerancing_jax`.  Each
points users at the NumPy version that's already complete.

`__all__` is now 373 entries.  All 25 validation files green.

## What's new in 3.5.0

Major release covering three sessions of feature work and API
harmonization.

- **JAX-traceable ray trace** (`trace_jax`) gains aspheric Newton
  intersect (`jax.lax.fori_loop`), aperture clipping, and DOE order
  kicks.  End-to-end differentiable via `jax.grad`.

- **JAX paths for the LG aberration tensor**:
  `aberration_tensor_lg00_jax` and
  `propagate_modal_asymptotic_lg00_jax` give differentiable
  Strehl-amplitude evaluation.

- **Multiple Huygens Surface (MHS) framework** -- new
  `lumenairy.propagators.mhs` module with `HuygensSurface`,
  `MhsSubdomain`, `MhsPipeline`, four convenience builders, and
  `MhsPipeline.from_prescription(...)` one-call ASM -> lens -> ASM
  chain builder.  Storage hooks via `pipeline.run(checkpoint=...,
  store=path)`.

- **`Source` class** bundles `(E, dx, wavelength, source_point,
  name)` with chainable `.propagate(...)` returning another
  Source; class-method factories `Source.gaussian`,
  `Source.plane_wave`, `Source.point_source`,
  `Source.top_hat`, `Source.fiber_mode`.

- **Smarter `propagate(method='auto')`** -- the dispatcher now
  inspects surfaces for aspheric coefficients, DOE events, and
  hard apertures.  New `accuracy='fast' | 'balanced' |
  'accurate'` hint.  `'mhs'` joins
  `'asm'`/`'fresnel'`/`'gbd'`/`'hf'`/`'hfpi'`/`'maslov'`/`'asymptotic'`.

- **Unified `PropagationResult`** -- opt-in via
  `return_result=True` on `propagate()`,
  `MhsPipeline.run(...)`, and `propagate_through_system(...)`.
  Carries `.field`, `.dx`, `.wavelength`, `.method`, `.history`
  list of `(label, field, dx)` planes, and `.metadata`.  Tuple-
  unpacks for backward compat; `np.asarray()` falls through to
  the field.

- **`replay_run(filepath, ...)`** reads back every plane stored
  by an MHS / system run, returning a `PropagationResult` with
  `.history` populated.

- **Optimize layer extensions:**
  - `wave_propagator='real_lens' | 'gbd' | 'hf' | 'hfpi' |
    'asymptotic'` plumbs any modern propagator into the wave leg.
  - `WAVE_PROPAGATOR_REGISTRY` + `register_wave_propagator(name,
    fn)` for user-registered custom propagators.
  - `JaxMeritTerm(fn, build_args=lambda x: ...)` enables analytic
    JAX gradients flowing into SciPy via `jac='auto'`.

- **Unified `aberration_summary(prescription, wavelength, ...)`**
  returns Seidel + EFL/BFL + LG aberration tensor in one call;
  `differentiable=True` routes through the JAX path.

- **`__all__` reorganized into 10 user-journey tiers** -- 358
  entries grouped by build/propagate/trace/analyze/optimize/...
  -- no entries removed.

- **`examples/` directory** with five end-to-end runnable scripts
  demonstrating the core workflows.

```python
import lumenairy as la

# One-liner system propagation:
src = la.Source.gaussian(w0=200e-6, N=256, dx=4e-6, wavelength=1.31e-6)
out = src.propagate(method='asm', z=10e-3)

# Aberration analysis:
presc = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                         glass='N-BK7', aperture=6e-3)
print(la.format_aberration_summary(la.aberration_summary(presc, 1.31e-6)))

# JAX-differentiable design optimization:
import jax.numpy as jnp
term = la.JaxMeritTerm(
    fn=lambda R: jnp.abs((R - 30e-3) * 1e3),
    build_args=lambda x: (x[0],),
    real_part=True,
)
res = la.design_optimize(parameterization=..., merit_terms=[term],
                          jac='auto', ...)
```

See [`examples/`](examples/) for full runnable scripts.

## What's new in 3.4.0

Major release.

- **Multi-backend (NumPy / CuPy / JAX).** New `lumenairy.backend`
  subpackage provides namespace dispatch (`array_namespace`),
  array predicates, multi-backend FFT (preserves the long-standing
  pyFFTW > scipy.fft > numpy.fft hierarchy for NumPy + adds cuFFT
  for CuPy + JAX-XLA for JAX arrays), `RandomState` with
  per-backend RNG idioms, and scipy-special / linalg dispatch.
  JAX arrays are differentiable via ``jax.grad`` and JIT-compilable
  via ``jax.jit``.

- **New `lumenairy.propagators` subpackage** holding all
  diffraction propagators -- existing (`propagation`, `asymptotic`)
  and new (`hfpi`, `gbd`, `hf`, `dispatch`, `subaperture`).
  Existing import paths still work via thin re-export shims.

- **`hfpi`** -- Huygens-Fresnel Path Integration (Monte Carlo
  ray-based, handles cascaded diffraction).
- **`gbd`** -- Gaussian Beamlet Decomposition (deterministic
  ray-based, ~100x faster than HFPI on smooth systems).
- **`hf`** -- Van-Vleck-corrected deterministic HF.
- **`dispatch`** -- top-level ``propagate(E, ..., method='auto')``
  smart selector.
- **`subaperture`** -- patch decomposition utilities.

- **Bundle naming unification.** New `PathBundle` (HFPI) and
  `BeamletBundle` (GBD) share ``positions`` / ``directions``
  field names with the existing `RayBundle`.

- **`REFERENCES.txt`** at the top level consolidates every external
  citation; inline citations have been removed from source.

- **Import alias** is now ``import lumenairy as la`` throughout
  the docs and tests (previously ``as op``).

```python
import lumenairy as la
E_out = la.propagate(E_in, z=1e-3, wavelength=1.31e-6, dx=4e-6,
                     method='auto')
```

The library is now organised into eight thematic subpackages
(`backend`, `propagators`, `raytrace`, `elements`, `io`,
`analysis`, `sources`, `optimize`).  All previous top-level
import paths (``from lumenairy.propagation import X``,
``from lumenairy.lenses import Y``, etc.) continue to work via
thin re-export shims; no user-visible breakage.

## What's new in 3.3.3

- **`recommend_grid_for_prescription`** — design-time companion to
  `check_grid_vs_apertures` that returns a recommended `(N, dx)`
  given a prescription, wavelength, source waist, and (optionally)
  DOE order range / period / DOE-to-destination distance.  The
  recommendation rounds `N` to a power of two by default and
  round-trips cleanly with `check_grid_vs_apertures` (no warnings
  fired at the recommended grid).

- **`scale_prescription(prescription, factor)`** — single-factor
  geometric self-similarity utility.  Scales radii, thicknesses,
  aperture / semi-diameters, object distance, aspheric coefficients
  (`A_n / factor**(n-1)`), and coord-break decenters; preserves
  conics, glasses, tilts, and wavelength.  F-number, NA, and
  magnification are invariant.  Useful for swapping between mm and
  m scales without re-deriving every surface.

- **Endpoint-anchored Chebyshev nodes** in
  `fit_canonical_polynomials` — opt-in `endpoint_anchored=True`
  kwarg rescales the Chebyshev-Gauss roots so the outermost node
  sits exactly on the [-1, 1] boundary, lowering max error when
  the fit support is the full source / pupil box.  Defaults to
  `False` to preserve existing numerics.

- **Documented `integration_method='local_quadrature'`** in
  `apply_real_lens_maslov` — per-pixel v2-disk integration via
  Newton + Hessian (more rigorous than a global linear fit at the
  cost of one Newton solve per output pixel).  Already existed in
  the library; the docstring now fully covers all three integration
  modes (`'quadrature'`, `'local_quadrature'`, `'stationary_phase'`).

## What's new in 3.3.2

- **Embedded grating diffraction in `trace()` and
  `fit_canonical_polynomials`** — new `surface_diffraction` kwarg
  pins a DOE / grating order at a specific surface inside a
  sequential prescription.  Required to do LG-aberration-tensor or
  asymptotic-propagator analysis at non-zero diffraction orders
  (Dammann splitter corner orders, etc.).  Applies the angular kick
  AND adds the DOE's linear-phase OPL contribution `m * lambda *
  (x, y) / period` at the surface, so per-emitter (0, 0) pistons are
  correct even at corner orders.

## What's new in 3.3.1

- **Pre-flight grid-vs-aperture check** — `apply_real_lens`,
  `apply_real_lens_traced`, and `apply_real_lens_maslov` now
  inspect every surface's `semi_diameter` against the simulation
  grid's half-extent (`N*dx/2`) and emit a `UserWarning` if any
  surface exceeds the grid.  This catches the silent-energy-loss
  case where the lens itself would have transmitted past the
  grid edge but the simulation truncates at `N*dx/2`, which
  otherwise manifests downstream as a uniform inward centroid
  bias and missing power.  A new public helper
  `check_grid_vs_apertures(prescription, N, dx,
  safety_factor=1.0)` returns the offending surfaces explicitly
  for use in pre-flight scripts.  Warning fires once per call
  site (Python's default warning filter dedups by source line).

- **Quadoa Optikos `.qos` import/export (best-effort)** —
  `export_quadoa_qos` / `load_quadoa_qos` add round-trip support
  for a Quadoa-Optikos-style JSON system file.  The schema
  (version `QUADOA_SCHEMA_VERSION = '1.0'`) captures every field
  a lumenairy prescription holds — radii (incl. biconic Y),
  conics, asphere coefficients (per axis), glasses, thicknesses,
  semi-diameters, aperture, stop index, wavelength, units.
  Round-trips lossless inside lumenairy; external Quadoa
  readability is unverified pending a reference `.qos`.  The
  library now has full prescription I/O for Zemax (`.zmx`,
  `.txt`), Code V (`.seq`), and Quadoa Optikos (`.qos`).

## What's new in 3.3.0

A new module **`lumenairy.asymptotic`** implementing the closed-form
phase-space (Maslov) diffraction propagator and Laguerre-Gaussian
aberration tensor.  This is the
"missing middle tier" between expensive wave-leg merits
(`StrehlMerit`, `RMSWavefrontMerit`) and cheap ray-leg-only merits
(`SphericalSeidelMerit`, `FocalLengthMerit`):  wave-leg-faithful
quantities (the named aberrations the diffraction integral sees) at
ray-leg-only evaluation cost.

- **`fit_canonical_polynomials`** — Trace a 4-D Chebyshev-node grid
  through any prescription, fit Phi(s2, v2) and s1(s2, v2) as
  4-variable Chebyshev tensor-product polynomials, expose analytic
  gradient evaluation.  Sub-microwave residual on refractive
  systems.

- **`aberration_tensor`** — Evaluate the LG aberration tensor
  T_{k;n,m} at a chief-ray image point.  Indices (p, ell)
  correspond directly to classical Seidel/Zernike aberrations:
  (1, 0) is defocus, (2, 0) is primary spherical, (1, +-1) is coma,
  (0, +-2) is astigmatism, etc.  Closed-form Wick-contracted
  Gaussian moment, no quadrature.

- **`propagate_modal_asymptotic`** — Closed-form leading-order
  asymptotic propagator on a 2-D output grid.  Reduces to Collins'
  ABCD in the source-dominated limit and Fourier-of-pupil in the
  pupil-dominated limit; interpolates smoothly with no caustic
  pathology.  ~10**3-10**4 times faster per pixel than direct
  quadrature.

- **`LGAberrationMerit`** — New `MeritTerm` subclass that drops
  into `design_optimize`.  Specifies named aberration channels to
  suppress (defocus, spherical, coma, ...); evaluates the closed-
  form tensor in milliseconds per merit call.  No wave leg required.

  ```python
  merit = la.LGAberrationMerit(
      targets={(2, 0): 1.0,    # primary spherical
               (1, 1): 1.0,    # coma (sin)
               (1, -1): 1.0,   # coma (cos)
               (0, 2): 1.0, (0, -2): 1.0},   # astigmatism
      field_points=[(0.0, 0.0), (5e-3, 0.0), (0.0, 5e-3)],
  )
  ```

- **LG / HG basis utilities** — `lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`,
  `decompose_hg`, `lg_seidel_label`.

- **Wick moment utilities** — `gaussian_moment_2d`,
  `gaussian_moment_table_2d`.  Closed-form 2-D Gaussian moments for
  complex-symmetric covariances.

> Validation:  32 new physics-faithful tests in
> `validation/test_asymptotic.py` (LG orthonormality to 1e-14,
> Wick / Isserlis identities, fit round-trip, modal propagator PSF
> peak location, end-to-end LGAberrationMerit).  Full existing test
> suite re-runs green:  no regressions.  No breaking API changes.

## What's new in 3.2.15

- **`apply_doe_phase_traced`** — new ray-trace primitive for splitting
  a `RayBundle` at a thin grating / DOE plane into one or more
  diffraction orders.  Applies the grating-equation direction-cosine
  shift `L_new = L + m_x * lambda / period_x` (and same on y) per
  ray, recomputes `N` from the unit-norm constraint, and flags
  evanescent orders (`L'^2 + M'^2 > 1`) `alive=False` with a new
  `RAY_EVANESCENT = 5` error code.  Two calling conventions: scalar
  orders return a same-shape bundle; 1-D order arrays return a
  bundle replicated `n_orders * n_rays` in order-major layout, ready
  for a single `trace()` call through the post-grating optics.  Use
  case: ray-trace through a Dammann splitter or any thin grating
  in a sequential prescription.

  > Validation: 6 new tests in `test_raytrace.py` cover zero-order
  > no-op, the grating equation, unit-norm preservation, evanescent
  > flagging, order-major layout, and free-space round-trip;
  > 32/32 raytrace + 17/17 optimize tests pass.

## What's new in 3.2.14

A focused performance pass on the highest-traffic paths.  All
behaviour preserved (full validation suite still 16/16 PASS,
298 assertions); changes are transparent caches or opt-in toggles.

- **ASM transfer-function `H` cache** keyed by
  `(N_y, N_x, dy, dx, λ, z, bandlimit, dtype)`.  Repeat propagations
  at the same geometry skip the chunked kernel build entirely.
  Measured **1.5× speedup at N=2048** on cache-hit (the H build is
  ~30-50% of total ASM call time on 2k+ grids).  Tunables:
  `set_asm_cache_size(...)`, `clear_asm_caches()`.

- **Frequency-grid + band-limit caches** complement the H cache so
  even on H-miss the kernel reconstruction skips spatial-frequency
  vector recomputation.

- **Multi-slot pyFFTW plan cache** (was single-slot per direction
  → 8 entries by default).  No more thrashing when calls oscillate
  between two shapes (JonesField stacking, Maslov inner work, batched
  vs scalar grids).  Tunable: `set_fft_plan_cache_size(n)`.

- **Single-precision (complex64) toggles**: `set_default_complex_dtype(np.complex64)`
  flips the library default for fields allocated when callers pass
  real-valued inputs.  All propagators preserve the caller's complex
  dtype; the kernel-phase mod-2π folding keeps single-precision ASM
  accurate at the float32 noise floor.  **2.18× speedup** on N=2048
  ASM, ~2× memory headroom.

- **`angular_spectrum_propagate_batch(E_3d, ...)`** runs a stack of
  fields `(B, Ny, Nx)` through one fused FFT pair across the
  trailing two axes.  `JonesField.propagate` stacks `[Ex, Ey]` and
  routes through it for grids ≥ 512 (below that, dispatch overhead
  exceeds the benefit and the H cache already serves the second
  component for free).

- **Numba-fused aspheric-sum kernel** in `surface_sag_general`.  The
  legacy aspheric loop allocated one fresh N×N array per coefficient
  term; the new path walks `h_sq` once and accumulates all terms in
  a single threaded pass.  **4.75× speedup** at N=2048 with 5
  aspheric coefficients.  Pure spheres unaffected; CuPy stays on
  the legacy path.

- **Memory-leaner refraction step** in `apply_real_lens` (Fresnel /
  slant_correction branches): gradient and intermediate arrays
  freed eagerly.  Drops peak transient memory at N=8192 from ~5 GB
  to ~1.5 GB.  Math unchanged.

## What's new in 3.2.13

- **Validation suite expanded** by ~70 new physics + interop test
  cases.  16 files, 298 Harness assertions across topic suites,
  all PASS.  Highlights: ASM linearity / reciprocity / D4σ growth;
  thin-vs-real lens interop; doublet ABCD-EFL ↔ paraxial focus
  agreement; Strehl-Maréchal; Parseval; Zernike linearity; Malus
  + crossed polarizers + circular S₃; Köhler imaging smoke;
  Cauchy-Schwarz on mutual coherence; Keplerian |M|=f₁/f₂;
  end-to-end singlet wave-vs-trace consistency;
  `propagate_through_system` matches manual `apply_real_lens`+ASM.

## What's new in 3.2.10 - 3.2.12

These were **UI-focused releases** that did not change core
library behaviour.  See `Optical_Propagation_Library_UI/CHANGELOG.md`
for the GUI-specific feature additions (workspace tabs, Welcome
dock, embedded Python REPL, persistent status-bar metrics,
drag-and-drop file open, keyboard shortcuts for workspaces, compact
mode, etc.).  Core API unchanged.

## What's new in 3.2.3

- **`wave_propagator='fresnel'` and `wave_propagator='rayleigh_sommerfeld'`**
  added to `apply_real_lens` (and threaded through `apply_real_lens_traced`).
  Together with the pre-existing `'asm'` (default) and `'sas'`, the
  through-glass propagator can now be switched to any of the four
  physically-sensible choices.  ASM remains the right tool for the mm-scale
  glass distances typical of lenses; the others are exposed for research
  and pipelines that want one propagator used consistently throughout.
  Fresnel and SAS resample back to the input `dx` automatically.  RS
  preserves pitch and agrees with ASM to ~1e-13 at typical through-glass
  distances.  Unknown `wave_propagator` values now raise `ValueError`.

## What's new in 3.2.2

- **`lens_maslov.py` retired; `apply_real_lens_maslov` now lives inside
  `lenses.py`** alongside `apply_real_lens` and `apply_real_lens_traced`.
  Was conceptually a third real-lens wave-optics pipeline all along,
  not a separate subsystem.  Public API unchanged:
  `op.apply_real_lens_maslov` and
  `from lumenairy.lenses import apply_real_lens_maslov`
  both still work.  Only the legacy
  `from lumenairy.lens_maslov import ...` path broke; nothing
  in the library or validation suite used it.

## What's new in 3.2.1

- **SAS integration hooks** — the Scalable Angular Spectrum propagator
  added in 3.2.0 is now a first-class peer of ASM/Fresnel in three
  more places:

    * `propagate_through_system({'type': 'propagate', ..., 'method': 'sas'})`
      — SAS alongside `'asm'` and `'fresnel'` as a per-element method.
      Pipeline auto-resamples back to `dx` so downstream elements keep
      their coordinates.
    * `apply_real_lens(..., wave_propagator='sas')` (+ forwarded through
      `apply_real_lens_traced`) — swap the through-glass propagator.
    * `JonesField.sas_propagate(z, wavelength, pad=2,
      skip_final_phase=False)` — polarization-aware SAS wrapper that
      applies the scalar SAS kernel to `Ex` and `Ey` and updates
      `self.dx` / `self.dy` to the new output pitch.

## What's new in 3.2.0

- **Scalable Angular Spectrum (SAS) propagator**
  (`scalable_angular_spectrum_propagate`).  Three-FFT kernel from
  Heintzmann-Loetgering-Wechsler 2023: ASM-minus-Fresnel precompensation
  phase + band-limit filter + Fresnel chirp + FFT + optional final
  quadratic phase.  Output pitch is `lambda*z/(pad*N*dx)` — a zoom-out
  that avoids the impractical-N trap of plain ASM at long z.  The
  paper's closed-form `z_limit` check warns when exceeded.  Includes a
  Fresnel-style `1/(i·λ·z)·dx²` amplitude prefactor for power
  conservation (the reference PyTorch notebook is amplitude-agnostic).
  Validated against `fresnel_propagate` / `fraunhofer_propagate` at
  moderate / far-field z in their respective limits.

- **CODE V `.seq` import/export**
  (`export_codev_seq` + `load_codev_seq`).  Canonical CODE V sequence
  syntax (`LEN NEW` / `DIM M|MM|IN` / `WL` / `S<i>` / `RDY` / `THI` /
  `GLA` / `CON` / `STO` / `APE`).  Bit-exact round-trip for radii,
  thicknesses, conic, glass, stop index, and aperture.

- **BSDF surface scatter model** (new module `bsdf.py`) with
  `LambertianBSDF`, `GaussianBSDF`, `HarveyShackBSDF`.  Common
  interface: `evaluate(inc, scat)`, `sample(inc, n, rng)`,
  `total_integrated_scatter()`.  Attached to `Surface` via a new
  `bsdf` field.  Helper `sample_scatter_rays(surface, incident,
  n_per_ray)` spawns a `RayBundle` of scattered rays for Monte Carlo
  stray-light propagation through the rest of a system.

- **Jones pupil spatial-map visualization**
  (`plot_jones_pupil` + `compute_jones_pupil`).
  `compute_jones_pupil(apply_fn, N, dx, wavelength)` probes a
  polarization-capable system with orthogonal x/y plane-wave inputs
  and returns the full `(Ny, Nx, 2, 2)` Jones matrix.
  `plot_jones_pupil(J, ...)` produces the canonical 2x4 grid
  (amplitude + phase for each of Jxx/Jxy/Jyx/Jyy) with phase masked
  below an amplitude threshold.

## What's new in 3.1.11

- **Stop-aware `seidel_coefficients`** — new `stop_index` and
  `field_angle` kwargs.  When the declared stop is not at surface 0,
  the chief ray's initial conditions are now derived from the pre-stop
  ABCD so that `y_chief = 0` at the stop by construction.  Default
  uses `find_stop` (which falls back to surface 0 when no surface is
  flagged `is_stop=True`).
- **`refocus(result, delta_z, wavelength=None)`** — closed-form
  image-space transfer of a traced bundle.  `through_focus_rms`
  uses it internally for a 5-20x speedup on focus sweeps.
- **`find_stop`, `compute_pupils`, `find_lenses`, `lens_abcd`,
  `LensInfo`, `PupilInfo`** — new stop / pupil / per-lens paraxial
  helpers.  `lens_abcd` accepts a prescription dict, surface-list
  slice, single `Surface`, or a `LensInfo` (with the original
  `surfaces` passed as a kwarg, for re-analysis at a different
  wavelength).
- **GPU path for `apply_real_lens`** (opt-in `use_gpu=True`).
  `apply_real_lens_traced` forwards `amp_use_gpu` to its two internal
  `apply_real_lens` calls.

## What's new in 3.1.10

- **`apply_real_lens(use_gpu=False)`** — opt-in CuPy backend for
  the full phase-screen + ASM-through-glass pipeline.  Default is
  ``False`` (unchanged CPU behaviour).  When enabled, per-surface
  sag arrays, phase screens, and ASM propagation all run on GPU.
- **`apply_real_lens_traced(amp_use_gpu=False)`** — new kwarg
  pipes ``use_gpu`` through to the internal ``apply_real_lens``
  calls that build the amp + amp(pw) arrays.  The rest of the
  traced pipeline (ray trace, Newton, assembly) stays CPU;
  results are pulled back automatically at the amp-block exit.
  Combines cleanly with the existing ``use_gpu=True`` kwarg for
  the Newton inversion.
- **`surface_sag_general` / `surface_sag_biconic`** now array-API
  polymorphic (accept NumPy or CuPy inputs transparently).

## What's new in 3.1.9

- **`lens_abcd(lens, wavelength)`** + **`find_lenses(surfaces,
  wavelength)`** — paraxial characterisation of individual lens
  elements.  Accepts prescription dicts, surface-list slices, or
  single surfaces.  Auto-detects cemented-doublet grouping.
  Returns EFL, BFL, FFL, principal planes, and the underlying
  ABCD for composition.
- **`compute_pupils(surfaces, wavelength)`** — paraxial entrance
  / exit pupil positions and radii, from the stop surface's ABCD
  sub-system images.  Foundation for future chief-ray aiming
  and rigorous reference-sphere OPD.
- **`RayBundle.error_code`** per-ray diagnostic field recording
  why each dead ray was killed (`RAY_TIR`, `RAY_APERTURE`, etc.).
  `trace_summary` now prints the breakdown.
- **Glass indices pre-resolved** once per `trace()` call; tiny
  per-call speedup, meaningful at high repeated-trace counts
  (aim iteration, focus sweeps, optimisation loops).

## What's new in 3.1.8

- **`trace(output_filter='last')`** eliminates per-surface
  ``RayBundle.copy()`` allocations when only the final bundle
  is consumed.  Saves ~1.4 GB per `apply_real_lens_traced`
  call on a 6-surface doublet at N=32768.  Wired into
  `apply_real_lens_traced` automatically.
- **`refocus(result, delta_z)`** — closed-form image-space
  transfer of a traced bundle.  `through_focus_rms` rewritten
  to use it, giving ~5-20x speedup on focus sweeps.
- **`is_stop` field on `Surface`** + **`find_stop(surfaces)`**
  helper.  Aperture-stop surface can be explicitly flagged
  (Zemax loaders populate it from the STOP keyword); helpers
  for pupil calculation and chief-ray aiming are the natural
  next steps.
- Seidel `S4` (Petzval) double-assignment fixed; dead-code in
  `_paraxial_trace` removed.  Numerical output unchanged.

## What's new in 3.1.7

- **`apply_real_lens_maslov`** — a third thick-lens propagator
  complementing `apply_real_lens` and `apply_real_lens_traced`.  Fits
  a 4-variable Chebyshev tensor-product polynomial to the ray-traced
  canonical map (`s1(s2, v2)`, `OPD(s2, v2)`) and evaluates the
  Maslov phase-space diffraction integral by stationary-phase
  (recommended, closed-form per-pixel), uniform Tukey quadrature
  (extended-source regime), or Hessian-oriented local quadrature
  (asymptotic corrections beyond leading stationary phase).
  Caustic-safe by construction, no critical-sampling constraint,
  analytically differentiable w.r.t. the polynomial coefficients.
  See `CHANGELOG.md` for validation numbers and regime guidance.

- **`apply_real_lens_traced` speedup kwargs** (all opt-in, default
  physics unchanged):

    * `fast_analytic_phase=True` — skip the full ASM-through-glass
      reference-phase pass in favour of a per-pixel sum of sag phase
      screens.  ~25 % wall-time savings when `parallel_amp=False`.
      Introduces <10 nm OPL phase error on typical refractive
      prescriptions.

    * `newton_fit='polynomial'` (new default) or `'spline'` — the
      polynomial path uses a 2-D Chebyshev tensor-product fit
      instead of `scipy.interpolate.RectBivariateSpline` for the
      entrance->exit map.  Implements combined value+gradient
      evaluation and an optional Numba `@njit(parallel=True)`
      fastpath: **~12x faster than the spline path on the Newton
      hot loop** (4M-sample isolated benchmark).  For smooth
      refractive systems (all Seidel and higher-order aberrations
      are polynomials), same or better accuracy with closed-form
      analytic derivatives; flip back to `'spline'` for high-order
      freeforms or sharp non-polynomial surface features.

    * `use_gpu=True` — dispatches the polynomial evaluator and
      Newton inversion to GPU via CuPy (amp/amp(pw)/ray-trace/final
      assembly stay CPU-only).  Requires
      `newton_fit='polynomial'` and cupy installed.  Output is
      bit-equivalent to CPU (0 % RMS error).  Modest absolute
      speedup at typical workloads (~1.4x vs numba CPU on 1-4M
      Newton samples); best for iterated design-optimisation
      workflows or very large grids.

- **Optional `numba` dependency** for the polynomial-Newton CPU
  fastpath.  `requirements.txt` now lists it under "optional"; the
  library falls back to a pure-NumPy path when numba is missing.

- **`apply_real_lens_traced` default `ray_subsample` bumped 1 → 8.**
  At typical production grid sizes (N=2048 and above) this gives
  hundreds to thousands of spline / polynomial samples across every
  lens aperture — far above the internal safety floor.  Small-grid
  users who would drop below 32 samples across aperture get a clear
  error message from the existing `on_undersample='error'` guardrail.

## What's new in 3.1.6

- **Zarr storage reliability fix for Windows + Python 3.14.**
  ``append_plane`` and ``write_sim_metadata`` no longer raise
  ``FileExistsError`` when reopening an existing zarr store.  See
  ``CHANGELOG.md`` for the root-cause breakdown; no API change for
  callers.

## What's new in 3.1.5

- **Zemax loaders preserve object-space distance.**
  `load_zmx_prescription` and `load_zemax_prescription_txt` now
  return a new `object_distance` key on the prescription dict,
  computed as the sum of `DISZ` values from the STOP (or SURF 0 if
  no STOP) up to the first active refractive surface.  This
  recovers design-intended obj-space geometry (coordinate breaks,
  field-reference planes, MLA mount surfaces, etc.) that earlier
  loader versions dropped.  Wave-optics driver scripts should
  propagate their source field by this distance before invoking the
  first lens operator; failing to do so collapses the obj-space
  geometry and produces a defocus-like blur at the image plane
  proportional to the dropped distance (observed on the Design 51
  .zmx: 96.67 mm of dropped air gap → ~235 µm defocus blur at the
  metasurface plane when the distance was not re-injected).

## What's new in 3.1.4

- **`apply_real_lens_traced(..., tilt_aware_rays=...)` default changed
  from `True` to `False`.**  The "Tier 1 input-aware ray launch"
  added in 3.1.2 mixed reference frames between the traced OPL and
  the plane-wave-reference analytic lens phase used in the
  `preserve_input_phase=True` subtraction.  On single-mode
  plane-wave-like inputs the mismatch was small; on multi-mode
  inputs (post-DOE fields, compound superpositions) it produced
  materially wrong output fields.  The plane-wave launch
  (`tilt_aware_rays=False`) is reference-consistent and correct for
  any input the wave model can represent.  See `CHANGELOG.md` for
  the full reasoning.
- **Paraxial-magnification Newton initial guess**: measured from the
  central finite-difference slope of the already-computed forward
  map (zero extra compute) instead of the hard-coded 1.10 multiplier.
  For compound-system callers this saves several Newton iterations
  per pixel; for singlets the improvement is marginal but the guess
  is always at least as good as before.
- **`inversion_method='backward_trace'` opt-in** on
  `apply_real_lens_traced` (experimental).  Replaces the
  forward-trace + Newton-spline-inversion with a direct backward
  ray trace from a coarse subsample of the exit grid through a
  reversed prescription.  Validated to agree with the Newton path
  to sub-pm on single-ray tests and ~30 nm OPD RMS end-to-end at
  N=1024 with a ~3x speedup.  Default stays `'newton'` while the
  backward path is validated on a wider set of prescriptions.

## What's new in 3.1.3

- **Multi-mode-safe `_sample_local_tilts`**: amplitude-weighted
  Gaussian smoothing of the tilt field (new `smooth_sigma_px` kwarg,
  default 4) before clipping, so post-DOE / interferometric inputs
  to `apply_real_lens_traced` degenerate gracefully to a collimated
  launch instead of injecting aliased per-pixel tilts that collapse
  the output field at large N.
- **Complex-dtype flexibility**: `apply_real_lens`, `apply_real_lens_traced`,
  `apply_mirror`, and `angular_spectrum_propagate` preserve the
  caller's complex dtype (complex128 or complex64) end-to-end.
  Kernel-phase and per-surface phase screens always compute in
  float64 + modulo-2-pi reduction before casting, so complex64 mode
  avoids the ~0.02-rad-per-Fourier-pixel precision floor that would
  otherwise swamp large-z ASM propagations.  New top-level export
  `DEFAULT_COMPLEX_DTYPE`.
- **FFT backend refresh**: pyFFTW now uses a single-slot per-direction
  plan cache with in-place aligned buffers (no more 30 s TTL alloc
  churn) and a per-plan `threading.Lock` so parallel callers share
  the cache safely.  Clean `reset_fft_backend()` support.
- **Parallel amp + amp(pw)** (new `parallel_amp=True` default) runs
  the two internal `apply_real_lens` calls inside `apply_real_lens_traced`
  on a `ThreadPoolExecutor`.  FFT-serialised, non-FFT work
  overlapped.  Auto-disables when available RAM is tight
  (`parallel_amp_min_free_gb`).
- **Amplitude-masked Newton** (`newton_amp_mask_rel=1e-4`): skip
  coarse-grid Newton pixels where the analytic amplitude is below
  threshold, since they'd be multiplied by ~zero in the final
  assembly anyway.  Big speedup on post-DOE / sparse fields; mask
  self-disables on dense fields.
- **Numexpr-fused phase screen in `apply_real_lens`** (optional
  dependency `[perf]`): `ne.evaluate('E * exp(-1j*k0*opd)', out=E)`
  eliminates ~50 GB of complex128 intermediates per surface at
  N=32768 and threads the operation.  Numpy fallback preserved.
- **Decenter-aliased entrance grids** in `apply_real_lens`: `Xs, Ys,
  h_sq` alias the axis-centred grids when decenter is zero (the
  common case), saving three float64 NxN allocations per surface.
- New top-level export `NUMEXPR_AVAILABLE` for runners that want to
  gate tunables on the fast path being importable.

## What's new in 3.1

- **Progress hooks** (`progress.py`) — optional `progress=callback` kwarg
  on `apply_real_lens`, `apply_real_lens_traced`, and
  `propagate_through_system`.  Drive a progress bar from any script or
  GUI; callback signature is `(stage, fraction, message)`.
  `ProgressScaler` lets long pipelines nest sub-tasks inside a parent
  budget.
- **`'real_lens_traced'` element type** for `propagate_through_system`
  so the hybrid wave/ray lens model is reachable through the unified
  element-list API.
- **Codegen promoted to public API** — `generate_simulation_script`,
  `generate_script_from_zmx`, and `generate_script_from_txt` are now
  exported from the top-level `lumenairy` namespace.
- **`remove_wavefront_modes` accepts `weights=`** for intensity-weighted
  piston / tilt / defocus fits; crucial on vignetted or annular pupils.
- **Freeform sags ray-traceable** — `Surface.freeform` plumbed through
  `_surface_sag_xy`, `_surface_sag_derivatives_xy`, and
  `surfaces_from_prescription` so XY-polynomial / Zernike / Chebyshev
  freeforms work in both wave and ray paths.
- **`make_singlet` / `make_doublet`** always emit biconic keys (set to
  `None`) for diff-friendly, round-trippable prescriptions.
- **`optical_table.py` + `.html` removed** — the bundled HTML simulator
  was unreferenced and its launcher functions were never exported.

## What's new in 3.0

- **`apply_real_lens_traced`** — high-accuracy hybrid wave/ray lens model.
  Sub-nanometer OPD agreement with the geometric ray trace on cemented
  doublets and other multi-surface curved-interface systems.  Uses
  `RectBivariateSpline` over the entrance grid + vectorised Newton
  inversion of the entrance→exit mapping; orders of magnitude better than
  the analytic thin-element model where it matters.  Amplitude from
  `apply_real_lens` (full ASM-through-glass), phase from ray-traced OPL.
- **Exit-vertex OPL correction** — `apply_real_lens_traced` now transfers
  rays from the last surface's sag to the flat exit vertex plane before
  computing OPD, using the **signed** parametric distance (not absolute
  value) so both concave and convex rear surfaces are handled correctly.
  Previously, off-axis rays ended at `z = sag(h) ≠ 0` while on-axis rays
  ended at `z = 0`, injecting systematic defocus (43 % on doublets) or
  catastrophic sign errors (200,000× on negative meniscus lenses).
  Doublet focus error: 10 mm → **0.000 mm**.  Negative meniscus residual:
  33,742 nm → **0.17 nm**.
- **Critical raytrace OPL bookkeeping fix** — `_intersect_surface` now
  accounts for the small "vertex-plane → actual-sag-intersection" leg in
  the right medium.  Singlet wave-vs-geom residuals dropped 17×–130×;
  every consumer of `raytrace.trace` benefits automatically.
- **Biconic / cylindrical / toroidal surfaces** — `surface_sag_biconic`,
  `make_cylindrical`, `make_biconic`, plus `radius_y` / `conic_y` /
  `aspheric_coeffs_y` keys on any prescription surface.  All downstream
  consumers (ray tracer, OPD analysis, both `apply_real_lens` variants,
  Seidel, ABCD) handle anamorphic surfaces transparently.
- **Zernike decomposition** — `zernike_decompose(opd_map, dx, aperture,
  n_modes)` fits OSA-normalised Zernikes via Householder QR with column
  pivoting (numerically stable for high-order, partial-pupil cases).
  Round-trips to ~1e-12 precision.  Plus `zernike_reconstruct`,
  `zernike_polynomial`, OSA index helpers.
- **Hybrid wave/ray design optimizer** (`lumenairy.optimize`)
  — refine a lens prescription against geometric and/or wave-based
  merit terms (focal length, Seidel S₁, Strehl ratio, RMS wavefront,
  spot size, chromatic focal shift).  Wraps `scipy.optimize` (L-BFGS-B
  by default; `lm` routes through Householder-QR-based Levenberg-
  Marquardt).  Pure-geometric optimization runs sub-second; wave-based
  metrics scale with grid size.
- **OPD-extraction Nyquist tooling** — `check_opd_sampling()` helper +
  built-in `RuntimeWarning` in `wave_opd_1d` / `wave_opd_2d` when
  sampling near or below the Nyquist edge for the lens's converging
  wavefront.  Optional `f_ref` parameter divides out a reference sphere
  before unwrap for users who want coarser grids.
- **SciPy FFT default** — `USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1`
  by default.  All wave-propagation calls now multithreaded; 2-4× speedup
  with zero memory overhead.  pyFFTW is opt-in.
- **`slant_correction` default reverted to `False`** — empirical
  validation showed paraxial `(n2−n1)·sag` is equal-or-better for almost
  every test case because the angular-spectrum propagation between
  surfaces already encodes the obliquity.  Slant correction remains
  available as opt-in.
- **Validation suite** — `validation/real_lens_opd/` now compares three
  methods (paraxial / slant / ray-traced) on 21 reference lenses with
  matching Zemax LDE + `.zmx` exports for cross-verification.

## Overview

`lumenairy` provides a physically accurate, modular toolkit for
simulating free-space coherent optics. It handles everything from basic
Gaussian beam propagation to multi-surface real lens modeling with glass
dispersion, metasurface design, and full Jones-vector polarization.

Features are implemented with well-tested physics (verified against textbook
formulas and audited for sign conventions), SI units throughout, and optional
GPU / multi-threaded FFT acceleration.

## Key Features

### Propagation
- **Angular Spectrum Method (ASM)** — exact, band-limited, with rectangular
  anti-aliasing filter
- **Tilted / off-axis ASM** — for beams with a non-zero carrier angle
- **Single-FFT Fresnel** — paraxial, changes output grid spacing
- **Fraunhofer (far-field)** — simplest far-field computation
- **Rayleigh-Sommerfeld** — convolution with the free-space Green's function,
  exact near-field diffraction without band-limiting approximation
- **Scalable Angular Spectrum (SAS)** — Heintzmann-Loetgering-Wechsler 2023
  three-FFT kernel with variable output pitch (`λz/(pad·N·dx)`); the right
  tool when z is long enough that plain ASM needs an impractically large
  grid

### Lenses
- **Thin lens** (paraxial, non-paraxial, aplanatic, local-only)
- **Spherical singlet** — exact OPD through thick glass
- **Aspheric singlet** — conic + even polynomial coefficients
- **Multi-surface real lens** (`apply_real_lens`) — default fast wave
  model.  Split-step refraction with ASM between surfaces; optional
  Fresnel transmission, bulk absorption, slant correction, Seidel
  correction.
- **Hybrid wave/ray real lens** (`apply_real_lens_traced`) — per-pixel
  ray-traced OPL + wave amplitude envelope; sub-nm OPD on cemented
  doublets and other multi-surface curved-interface systems.
- **Phase-space / Maslov real lens** (`apply_real_lens_maslov`) —
  Chebyshev fit of the ray-traced canonical map + Maslov integral.
  Caustic-safe and differentiable; pair with
  `apply_real_lens_maslov_jax` for autodiff design optimisation.
- **Pluggable through-glass propagator** (`wave_propagator=` kwarg
  on `apply_real_lens` / `apply_real_lens_traced`): `'asm'`
  (default), `'sas'`, `'fresnel'`, `'rayleigh_sommerfeld'`.

  *See the [Appendix](#appendix-physics-references) for the
  underlying physics / accuracy trade-offs, or
  `help(la.apply_real_lens)` for the per-function decision guide.*
- **Biconic / cylindrical / toroidal** elements via `make_biconic` and
  `make_cylindrical` plus optional `radius_y`/`conic_y` keys on any
  prescription surface
- **GRIN rod lens** — gradient-index parabolic profile
- **Axicon** — conical lens (Bessel-beam generator)

### Geometric Ray Tracing
- **Sequential 3-D ray tracer** — vectorised Snell's law with exact
  conic/aspheric surface intersection (Newton iteration)
- **Surface types** — sphere, conic, aspheric (Zemax standard sag), flat, mirror
- **ABCD matrix** extraction — EFL, BFL, FFL from paraxial marginal ray
- **Seidel aberrations** — per-surface third-order coefficients (S1–S5)
- **Spot diagrams** — with RMS/GEO radius, Airy disc overlay
- **Ray fan plots** — transverse ray aberration vs normalised pupil
- **OPD analysis** — wavefront error fans
- **Through-focus** — RMS spot vs defocus with best-focus finder
- **Ray generators** — fans, grids, concentric rings, single rays
- **Diffraction-order shift** — `apply_doe_phase_traced` splits a ray
  bundle at a thin grating / DOE into one or more orders (single or
  array), with evanescent flagging via `RAY_EVANESCENT`
- **Prescription compatible** — same prescription dicts as `apply_real_lens`
- **System compatible** — `raytrace_system()` accepts the same element list
  as `propagate_through_system()` for instant wave-optics ↔ ray-optics switching

### Mirrors
- Flat or curved (spherical / conic) with optional aperture

### Apertures and Masks
- **Hard apertures** — circular, rectangular, annular
- **Soft apertures** — Gaussian
- **Arbitrary complex masks** — for SLMs, metasurfaces, custom DOEs

### Wavefront
- **Zernike aberrations on a pupil** — `apply_zernike_aberration` for
  generating wavefronts from Zernike coefficients
- **Zernike decomposition of OPD maps** — `zernike_decompose` /
  `zernike_reconstruct` using Householder QR with column pivoting
  (numerically stable, OSA-normalised, RMS coefficients in meters)
- **Zernike basis primitives** — `zernike_polynomial(n, m, rho, theta)`,
  `zernike_basis_matrix`, `zernike_index_to_nm`, `zernike_nm_to_index`
- **OPD extraction from wave fields** — `wave_opd_1d`, `wave_opd_2d`
  with Nyquist sampling warnings, optional reference-sphere subtraction
- **Sampling-rule helper** — `check_opd_sampling` reports the Nyquist
  margin and recommends grid sizing for clean OPD extraction
- **Low-order mode removal** — `remove_wavefront_modes` (piston / tilt /
  defocus least-squares fit and subtract)
- **Turbulence phase screens** — Kolmogorov and von Karman statistics

### Polarization (Jones calculus)
- **`JonesField` class** — wraps (Ex, Ey) with all standard propagators
  (ASM, Fresnel, Fraunhofer, tilted ASM, SAS as of 3.2.1) and all
  non-polarizing element methods (thin/spherical/real lens, aperture,
  mirror, mask)
- **Polarization elements** — polarizers, waveplates, rotators, arbitrary Jones matrices
- **Polarized sources** — linear, circular, elliptical
- **Analysis** — Stokes parameters, degree of polarization, polarization ellipse
- **Jones-pupil spatial map** (3.2.0) — `compute_jones_pupil(apply_fn, ...)`
  probes a system with orthogonal x/y inputs to extract the full 2×2
  exit-pupil Jones matrix, and `plot_jones_pupil` renders it as the
  canonical 2×4 amplitude + phase grid

### Beam Sources
- Fundamental Gaussian (TEM00)
- Hermite-Gauss modes (HG_{mn})
- Laguerre-Gauss modes (LG_{pl}) with OAM
- **Tilted plane wave** — off-axis collimated source at arbitrary field angles
- **Point source** — diverging spherical wave from (x0, y0, z0)
- **Multi-field source generator** — batch-produce tilted plane waves for field analysis

### Beam Analysis
- Centroid (center of mass)
- D4sigma (ISO 11146) beam diameter
- Power-in-bucket (circular or rectangular)
- Strehl ratio
- PSF, OTF, MTF (including radial MTF profiles)
- Sampling-condition diagnostics
- **Chromatic focal shift** — per-wavelength EFL/BFL and axial colour PV
- **Polychromatic Strehl** — weighted Strehl average across wavelengths

### High-NA Vector Diffraction (Richards-Wolf)
- **`richards_wolf_focus`** — compute (Ex, Ey, Ez) vectorial focal field
  from a scalar or Jones-vector pupil, handling NA > 0.5 where scalar
  diffraction breaks down (longitudinal Ez component)
- **`debye_wolf_psf`** — intensity PSF |E|^2 including all polarisation
  components
- Supports arbitrary input polarisation (x, y, circular, or custom Jones)
- Multi-z-plane evaluation for 3-D focal-volume tomography

### Partial Coherence / Extended-Source Imaging
- **`koehler_image`** — Koehler condenser illumination model: integrates
  coherent sub-images over the condenser NA to produce a partially-
  coherent image
- **`extended_source_image`** — arbitrary source angle distribution with
  per-direction weights
- **`mutual_coherence`** — compute Gamma(r1, r2) from a field ensemble

### Detector / Wavefront-Sensor Simulation
- **`apply_detector`** — pixel-integrate a field onto a detector grid
  with Poisson shot noise, Gaussian read noise, dark current, full-well
  saturation, and quantum efficiency
- **`shack_hartmann`** — simulate a Shack-Hartmann wavefront sensor:
  sub-aperture extraction, lenslet focusing, centroid detection, and
  wavefront reconstruction via slope integration

### Diffractive Optical Elements
- Periodic phase mask tiling (for DOEs, gratings)
- Microlens array
- **Dammann grating IFTA design** — generates uniform spot arrays

### Glass Catalog
- Refractive index lookup via the [refractiveindex.info](https://refractiveindex.info) database
- Includes Schott, Ohara, fused silica, silicon, CaF₂, etc.
- Cached material objects for fast repeated lookups

### Lens Prescriptions
- Build singlets and cemented doublets by glass name + geometry
- **Thorlabs catalog presets** — LA1050-C, LA1509-C, AC254-050-C, AC254-100-C, etc.
- **Zemax `.zmx` parser** — import real lens prescriptions from Zemax files
- **Zemax `.txt` parser** — import Zemax prescription-report text files
- **Zemax `.zmx` / `.txt` exporters** — round-trip designs back to Zemax
- **CODE V `.seq` import / export** — round-trips prescriptions through the
  canonical CODE V sequence syntax (units M/MM/IN, spherical + conic
  surfaces, stop flag, aperture; unknown directives skipped on import)

### Stray-Light / BSDF (3.2.0)
- **Three BSDF models** with a common `evaluate` / `sample` /
  `total_integrated_scatter` interface: `LambertianBSDF` (uniform
  diffuse), `GaussianBSDF` (small-angle lobe around specular),
  `HarveyShackBSDF` (three-parameter ABC model for polished
  microroughness with optional wavelength scaling)
- **`Surface.bsdf` field** — attach a BSDF to any sequential-system
  surface
- **`sample_scatter_rays(surface, incident, n_per_ray)`** — spawn a
  `RayBundle` of scattered rays sampled from the surface's BSDF for
  Monte Carlo stray-light propagation through the remainder of a
  system

### User Library
- **Persistent material catalog** — save custom glasses (fixed index or from
  refractiveindex.info) for reuse across sessions and scripts
- **Lens library** — save and load prescription dicts (singlets, doublets,
  custom designs, Thorlabs catalog lenses)
- **Phase mask library** — save mathematical expressions (e.g. spiral phase
  plates), pre-computed arrays, or glass-block definitions
- All stored as JSON in ``~/.lumenairy/library/``
- Saved materials auto-register in ``GLASS_REGISTRY`` on import

### Phase Retrieval
- **Gerchberg-Saxton** — phase-only CGH design between source and target
- **Error Reduction** — coherent diffractive imaging
- **Hybrid Input-Output (HIO)** — Fienup's feedback algorithm

### Prescription → Simulation Script (codegen)
- **`generate_simulation_script`** — turn a prescription dict into a
  standalone, runnable Python script that imports `lumenairy`,
  defines the prescription inline, builds an element list, and propagates
  a source through it. Useful for archiving a simulation alongside a
  design, sending a reproducible reference to a collaborator, or dropping
  the generated code into a Jupyter notebook as a starting point.
- **`generate_script_from_zmx`** / **`generate_script_from_txt`** —
  one-call path from a `.zmx` file or Zemax prescription text export
  straight to a simulation script.
- Styles: `'unrolled'` (one call per element, easy to edit) or
  `'system'` (single `propagate_through_system` call, compact).
- Toggle `include_analysis` and `include_plotting` to control how much
  post-propagation code is emitted.

### I/O
- CSV phase file read/write (with metadata header)
- FITS file read/write (optional, requires `astropy`)
- **HDF5 file read/write** (optional, requires `h5py`) — single fields,
  multi-plane propagation datasets, and polarized JonesFields with
  hierarchical groups, compression, and rich metadata attributes

### Plotting (optional, requires `matplotlib`)
- **Field visualizations** — intensity (log/linear), phase (with
  low-intensity masking), combined intensity + phase panels
- **Cross-sections** — 1D cuts through any axis with optional phase overlay
- **Multi-plane grids** — automatic layout for propagation simulation
  results with per-plane labels and z-positions
- **PSF / MTF** — 2D PSF plots and radial MTF profiles (with optional
  diffraction-limit overlay)
- **Polarization** — 4-panel Stokes parameter maps and polarization
  ellipse overlays on intensity images
- **Beam profile** — 1D intensity cross-section with D4σ markers and
  optional Gaussian fit

### System Propagation
- `propagate_through_system()` — pass a field through an ordered list of
  elements with one function call
- `raytrace_system()` — geometric ray-trace the **same** element list for
  quick cross-validation against wave-optics

### Through-focus / Tolerancing
- **`through_focus_scan`** — propagate the exit field across a range of
  axial planes and tabulate Strehl, peak intensity, D4σ spot, RMS
  radius, encircled energy at each plane
- **`find_best_focus`** — optimise the metric of choice across the scan
- **`single_plane_metrics`** — full set of beam metrics at a single z
- **`diffraction_limited_peak`** — reference for Strehl computations
- **`tolerancing_sweep`** — apply a list of `Perturbation` (decenter,
  tilt, form-error) and compare best-focus Strehl/spot for each
- **`monte_carlo_tolerancing`** — random perturbation draws from
  user-specified distributions, aggregate Strehl statistics

### Hybrid Wave/Ray Design Optimization (`lumenairy.optimize`)
- **`DesignParameterization`** — flat-vector ↔ prescription dict mapping
  with arbitrary path-based free variables and bounds
- **`MeritTerm`** building blocks: `FocalLengthMerit`,
  `BackFocalLengthMerit`, `SphericalSeidelMerit`, `StrehlMerit`,
  `RMSWavefrontMerit`, `SpotSizeMerit`, `ChromaticFocalShiftMerit`,
  **`LGAberrationMerit`** (closed-form named aberration suppression
  via the LG aberration tensor — see Phase-space asymptotic propagator)
- **`design_optimize`** — main driver wrapping `scipy.optimize`
  (L-BFGS-B / SLSQP / trust-constr / `lm`).  Wave leg only runs when a
  wave-based merit term needs it; pure-geometric optimization is
  sub-second for typical lenses
- `lm` method routes through `scipy.optimize.least_squares`, which uses
  Householder QR with column pivoting under the hood

### Phase-space asymptotic propagator (`lumenairy.asymptotic`)
- **`fit_canonical_polynomials`** — 4-variable Chebyshev tensor-product
  fit of `Phi(s2, v2)` and `s1(s2, v2)` from a ray-traced grid; sub-
  microwave residuals on refractive systems
- **`aberration_tensor`** — closed-form Laguerre-Gaussian aberration
  tensor whose indices correspond directly to classical Seidel/Zernike
  aberrations (defocus, spherical, coma, astigmatism, trefoil, ...)
- **`propagate_modal_asymptotic`** — closed-form leading-order Maslov
  propagator; Collins-ABCD in source-dominated limit, Fourier-of-pupil
  in pupil-dominated limit; ~10³-10⁴× faster per pixel than direct
  Maslov quadrature
- **`solve_envelope_stationary`** — Newton-solve the chief-ray
  envelope-stationary equation directly on the Chebyshev fit
- **LG / HG basis utilities** — `lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`, `decompose_hg`,
  `lg_seidel_label`
- **Wick moment utilities** — `gaussian_moment_2d`,
  `gaussian_moment_table_2d` for 2-D Gaussian moments under complex-
  symmetric covariance

## Installation

### Development install (editable)

From the project root (the `Optical_Propagation_Library/` directory),
install the package in editable mode:

```bash
cd Optical_Propagation_Library
pip install -e .
```

This makes `lumenairy` importable from anywhere and any edits
to the source files take effect immediately without reinstalling.

### Standard install

```bash
cd Optical_Propagation_Library
pip install .
```

### Manual (no install)

If you prefer not to install the package, you can place your scripts
next to the `Optical_Propagation_Library` directory and add it to
`sys.path`:

```python
import sys
sys.path.insert(0, 'Optical_Propagation_Library')
import lumenairy as la
```

### Usage

Once installed (or on `sys.path`):

```python
import lumenairy as la
# or
from lumenairy import angular_spectrum_propagate, JonesField
```

## Dependencies

Canonical source: `pyproject.toml` `[project.dependencies]` and
`[project.optional-dependencies]`.

### Required
- `numpy` — core numerics
- `scipy` — Zernike decomposition (Householder QR), `design_optimize`,
  multi-threaded FFTs (`USE_SCIPY_FFT = True`)
- `matplotlib` — all plotting utilities
- `psutil` — accurate memory detection (falls back to a 4 GB default
  without it)

### Optional (install via extras)
- `[glass]` — `refractiveindex>=1.0`.  Extended glass-catalog lookups
  (Hikari, Sumita, formula-3 CDGM).  v4.16.1 moved this from hard
  dep to an optional extras group; minimal installs transparently
  fall back to the bundled Sellmeier (and v4.16.2+ formula-3)
  evaluators for the 46+ glasses with bundled coefficients.
- `[fft]` — `pyfftw>=0.13`.  Multi-threaded CPU FFT; extra ~10–20%
  on top of SciPy FFT.
- `[gpu]` — `cupy>=11.0`.  GPU acceleration on NVIDIA CUDA or AMD
  ROCm (auto-detected; pass `use_gpu=True` to supported functions).
  Install the wheel that matches your hardware:
  `pip install cupy-cuda12x` (CUDA 12.x) or `pip install
  cupy-rocm-6-1` (ROCm 6.x).  See the
  [Installation wiki page](https://github.com/travaj24/LumenAiry/wiki/Installation#gpu-setup)
  for the full toolkit matrix and known-good GPU families.
- `[fits]` — `astropy>=5.0`.  FITS file I/O.
- `[hdf5]` — `h5py>=3.0` + `filelock>=3.0`.  HDF5 field storage.
- `[zarr]` — `zarr>=3.0` + `filelock>=3.0`.  Zarr storage backend
  (alternative to HDF5).
- `[jax]` — `jax>=0.4.20`.  Automatic differentiation (CPU).
- `[jax-gpu]` — `jax[cuda12]>=0.4.20`.  JAX with CUDA wheels.
- `[perf]` — `numexpr>=2.8`.  Fused phase-screen multiply
  (1.5–2× speedup at large N).
- `[numba]` — `numba>=0.58`.  JIT fastpath for traced-lens Newton-fit.
- `[multi_objective]` — `pymoo>=0.6`.  NSGA-II for
  `design_optimize_multi_objective`.
- `[gui]` — `PySide6` + `pyvista` + `pyvistaqt` + `h5py`.  LumenAiry
  Designer GUI.
- `[all]` — every optional accelerator bundled together.

Install the core (no glass extension):

```bash
pip install lumenairy
```

Add the glass-catalog extension (Hikari / Sumita / formula-3 CDGM):

```bash
pip install lumenairy[glass]
```

Kitchen-sink install:

```bash
pip install lumenairy[all]
```

## Quick start: which function should I use?

The library is wide -- pick a question and the table tells you where to start.

### I want to propagate a field through free space

| Situation | Use |
|---|---|
| Don't know which propagator to pick | `la.propagate(E, z=z, wavelength=wl, dx=dx)` -- smart auto-dispatch |
| Standard near/mid-field | `la.angular_spectrum_propagate(E, z, wl, dx)` |
| Need a specific output grid pitch (focal-plane zoom, MFT) | `la.fresnel_propagate_mft(...)` or `la.fraunhofer_propagate_mft(...)` |
| Long-z (z*lambda/(N*dx) > 1, plain ASM would need an oversized grid) | `la.scalable_angular_spectrum_propagate(...)` |
| Far-field | `la.fraunhofer_propagate(E, z, wl, dx)` |
| Through one ideal thin lens | `la.apply_thin_lens(E, f=f, wavelength=wl, dx=dx)` |

### I have a lens prescription (Zemax `.zmx`, Thorlabs catalog, or a dict)

| Situation | Use |
|---|---|
| Default fast wave model | `la.apply_real_lens(E, prescription=presc, wavelength=wl, dx=dx)` |
| Sub-nm OPD on cemented doublets or multi-surface curved-interface systems | `la.apply_real_lens_traced(E, prescription=presc, wavelength=wl, dx=dx)` |
| Inside a JAX-autodiff design loop, or near a caustic | `la.apply_real_lens_maslov(...)` / `la.apply_real_lens_maslov_jax(...)` |
| Just want a spot diagram | `la.trace_prescription(presc, wl, num_rings=8)` |
| Paraxial EFL / BFL / first-order data | `la.first_order_data(presc, wl)` |
| Element list shared between wave and ray paths | `la.propagate_through_system(...)` and `la.raytrace_system(...)` |

### I have a folded design (mirrors + coord-breaks)

| Situation | Use |
|---|---|
| Build world-frame surfaces from the prescription | `la.world_surfaces_from_prescription(presc)` |
| Ray-trace through folds | `la.trace_world(rays, wsurfs, wl)` |
| Paraxial image-plane position in world coordinates | `la.paraxial_focus_world(wsurfs, wl)` |

### I need to analyze the output field

| Situation | Use |
|---|---|
| PSF + MTF | `la.compute_psf(pupil, wl, f, dx)`, `la.compute_mtf(psf)` |
| Strehl ratio from a PSF (peak-ratio) | `la.strehl_ratio(E, E_ref, dx)` |
| Strehl from an RMS estimate (Marechal closed form) | `la.strehl_marechal(rms_waves)` |
| Strehl from a pupil (exact small-aberration) | `la.strehl_phase_integral(pupil)` |
| OPD / Zernike decomposition | `la.wave_opd_2d(...)`, `la.zernike_decompose(opd, dx, ap)` |
| Off-axis WFE at one field point | `la.eval_image_plane_wfe(presc, wl, field=(Hx, Hy))` |
| Full WFE-vs-field grid | `la.field_grid_wfe(presc, wl, field_max_m, n_field)` |
| Field-resolved distortion / sensitivity / per-surface footprint | `la.distortion_vs_field`, `la.sensitivity_ranking`, `la.footprint_per_surface` |
| Field-curvature / astigmatism vs field (real-ray) | `la.field_aberration_sweep(...)` |
| Off-axis Seidel coefficients | `la.seidel_field_sweep(surfaces, wl, field_heights)` |

### I want to optimize / tolerance a design

| Situation | Use |
|---|---|
| Hybrid wave/ray optimisation with composable merit terms | `la.design_optimize(parameterization, merit_terms)` |
| Tolerance sensitivity ranking | `la.sensitivity_ranking(merit_fn, x0)` |
| Through-focus / Monte-Carlo tolerancing | `la.through_focus_scan(...)`, `la.monte_carlo_tolerancing(...)` |

### Three minimal end-to-end examples

```python
# 1. Free-space propagation, smart dispatch -- works for any geometry.
import lumenairy as la
E, x, y = la.create_gaussian_beam(N=512, dx=2e-6, wavelength=1.31e-6, sigma=50e-6)
E_focus = la.angular_spectrum_propagate(E, z=1e-3, wavelength=1.31e-6, dx=2e-6)
print('centroid =', la.beam_centroid(E_focus, 2e-6))

# 2. A Thorlabs lens, end-to-end.
presc = la.thorlabs_lens('AC254-100-C')
E_out = la.apply_real_lens(E, prescription=presc, wavelength=1.31e-6, dx=2e-6)

# 3. Ray-trace the same lens for a spot RMS.
result = la.trace_prescription(presc, wavelength=1.31e-6, num_rings=8)
print(f'spot RMS = {la.spot_rms(result)[0]*1e6:.2f} um')
```

That's enough to start.  For longer recipes -- folded designs,
Zernike decomposition, optimisation loops, polychromatic PSFs,
HDF5 storage, plotting -- see the `Cookbook` section near the end
of this README.

## Cookbook

The deep recipe / use-case walkthrough is now at
[docs/cookbook.md](docs/cookbook.md).  Examples in
[`examples/`](examples/) are runnable scripts that cover most of
the same surface.

## Project Layout

```
Optical_Propagation_Library/            # project root
    README.md                            # this file
    LICENSE                              # MIT license
    CHANGELOG.md                         # version-by-version release notes
    pyproject.toml                       # build / install configuration
    requirements.txt                     # runtime dependencies
    lumenairy/                 # the importable package
        __init__.py                      # public API re-exports (272 symbols)
        propagation.py                   # ASM, tilted ASM, Fresnel, Fraunhofer,
                                         # Rayleigh-Sommerfeld, Scalable ASM
                                         # (SciPy-FFT default, multithreaded)
        lenses.py                        # Thin/thick/aspheric/real lenses +
                                         # apply_real_lens_traced (hybrid) +
                                         # apply_real_lens_maslov (phase-space),
                                         # surface_sag_general / _biconic,
                                         # cylindrical / GRIN / axicon
        glass.py                         # Glass catalog + refractiveindex.info
        coatings.py                      # Thin-film stack (TMM) + QW / broadband
                                         # AR coating designs
        bsdf.py                          # Surface-scatter BSDF models
                                         # (Lambertian, Gaussian, Harvey-Shack)
        elements.py                      # Mirrors, apertures, masks, Zernike
                                         # aberrations, turbulence phase screens
        sources.py                       # Gaussian, HG, LG, top-hat, annular,
                                         # Bessel, LED, fiber, tilted plane,
                                         # point source, multi-field generator
        freeform.py                      # XY-poly / Zernike / Chebyshev
                                         # freeform surface sags
        analysis.py                      # Centroid, D4σ, Strehl, PSF/OTF/MTF,
                                         # OPD extraction (1D/2D),
                                         # Zernike decomposition (QR-based),
                                         # check_opd_sampling, chromatic
        doe.py                           # Gratings, MLA, Dammann, FITS / phase
                                         # file I/O
        interferometry.py                # Interferogram synthesis + PSI
        phase_retrieval.py               # Gerchberg-Saxton, HIO, ER
        prescriptions.py                 # Singlet, doublet, biconic, cylindrical,
                                         # Thorlabs catalog, Zemax I/O, CODE V I/O
        system.py                        # Sequential element-list propagator
                                         # (accepts method='asm'|'fresnel'|'sas')
        raytrace.py                      # Geometric ray tracer (Snell, ABCD,
                                         # Seidel, pupils, find_lenses,
                                         # refocus, through_focus_rms, spot,
                                         # ray fan, OPD fan, error codes;
                                         # biconic-aware; OPL fix)
        through_focus.py                 # Strehl, best-focus, single-plane
                                         # metrics, tolerancing, MC analysis
        optimize.py                      # Hybrid wave/ray design optimizer
                                         # (DesignParameterization, 12+ MeritTerms,
                                         #  scipy.optimize wrapper; DE, basin-hop)
        vector_diffraction.py            # Richards-Wolf high-NA vectorial focus
        coherence.py                     # Koehler/extended-source partially-
                                         # coherent imaging, mutual coherence
        detector.py                      # Detector pixel model (shot/read noise,
                                         # dark current, full-well saturation),
                                         # Shack-Hartmann WFS simulator
        polarization.py                  # Jones vector / JonesField (with SAS
                                         # propagate), waveplates, Stokes
        ghost.py                         # Ghost-path analysis for a multi-
                                         # surface lens
        multiconfig.py                   # Multi-configuration / afocal
                                         # (Keplerian, beam expander)
        rcwa.py                          # 1-D rigorous coupled-wave analysis
        storage.py                       # Unified HDF5/Zarr auto-dispatch backend
        hdf5_io.py                       # Back-compat re-export shim for storage
        memory.py                        # RAM budget and batch-size helpers
        user_library.py                  # Persistent user material/lens/mask library
        codegen.py                       # Auto-generate sim scripts from Zemax
        plotting.py                      # Matplotlib plots (field, PSF, MTF,
                                         # Stokes, polarization ellipses,
                                         # Jones pupil 2×4 grid)
        progress.py                      # ProgressCallback + ProgressScaler
        _backends.py                     # CPU / thread helpers
    validation/                          # topic-based validation suite
        _harness.py                      # Shared Harness class
        run_all.py                       # discovers test_*.py + runs in
                                         # fresh subprocesses (per-file PASS/FAIL
                                         # + aggregate summary)
        test_propagation.py              # ASM, Fresnel, Fraunhofer, R-S, SAS,
                                         # tilted, sampling
        test_lenses.py                   # thin/real/biconic/cyl/asph/GRIN/axicon
                                         # + Maslov/traced, wave_propagator switch
        test_raytrace.py                 # Snell, ABCD, Seidel, pupils, refocus,
                                         # through_focus, spot/fans/off-axis
        test_analysis.py                 # Zernike, Strehl, MTF, Airy, Gaussian-
                                         # ABCD, OPD metrics, overlap invariance
        test_sources.py                  # 10 beam generators
        test_elements.py                 # apertures, mirror, turbulence, Zernike
        test_polarization.py             # HWP, QWP, Stokes, Jones, Jones pupil
        test_advanced_diffraction.py     # Richards-Wolf + Koehler + mutual
        test_optimize.py                 # all merits + L-BFGS / DE / basin-hop
        test_io.py                       # HDF5 / Zemax / CODE V / user_library
        test_features.py                 # coatings / DOE / RCWA / freeform /
                                         # ghost / interferometry / multi-config
                                         # + BSDF
        test_detector.py                 # detector / Shack-Hartmann / GS
        test_glass_tolerancing.py        # dispersion / achromat / tolerances
        test_integration.py              # API exports / compositions / memory
                                         # / plotting smoke
        test_subsample.py                # apply_real_lens_traced subsample
                                         # guardrail + ProcessPool + scaling
        test_validation_lens.py          # known-answer lens harness
        real_lens_opd/                   # 3-method OPD comparison sweep
            run_validation.py             # paraxial / slant / ray-traced
            results/                      # per-case PNGs + report.md/csv
            zemax_prescriptions/          # matching .txt/.zmx for cross-check
        through_focus_smoke/             # quick Strehl + tolerancing demo
```

## Conventions

- **Time dependence:** `exp(-i*omega*t)` throughout
- **Units:** SI meters for all spatial quantities
- **Sign convention for radii:** positive = center of curvature to the right
  of the surface (standard optics / Zemax convention)
- **Grid:** always square (N × N), centered at the origin

## Appendix: Physics references

Citation-heavy notes that don't belong in the main flow.  The
short version of "what the propagators are" is in the
[Quick start tables](#quick-start-which-function-should-i-use) at
the top of this file; everything below is here for users who want
the underlying physics.

The library is designed around the Angular Spectrum Method, which is exact
(within sampling limits) for free-space propagation. All band-limiting uses
the Matsushima-Shimobaba (2009) rectangular criterion for anti-aliasing.
The Scalable ASM (`scalable_angular_spectrum_propagate`) uses the
Heintzmann-Loetgering-Wechsler (2023) three-FFT kernel for long-z
propagation where the plain ASM grid would need to be impractically
large.

### Real-lens accuracy strategy

Two complementary models are provided:

- **`apply_real_lens` (analytic thin-element)** — split-step refraction
  with ASM between surfaces.  Each refracting surface is treated as a
  phase screen computed from the exact surface sag (conic + polynomial
  aspheric + biconic if specified), and the wave propagates through
  the glass between surfaces in the in-medium wavelength `lambda/n`.
  Captures diffraction during in-glass propagation; sub-100-nm RMS
  agreement with geometric ray trace on most singlets.  Has a hard
  accuracy ceiling on cemented doublets and other multi-surface
  curved-interface systems because the wave model treats each glass
  region as a vertex-to-vertex uniform slab while real rays cross
  interior interfaces at z = sag(h).
- **`apply_real_lens_traced` (hybrid wave/ray)** — bypasses that ceiling
  by computing the exit-pupil OPD from a geometric ray trace (per-pixel
  Newton inversion of the entrance→exit map via cubic splines) and
  combining with a wave-optics amplitude envelope.  Sub-nanometer OPD
  agreement with the geometric ray trace when properly sampled, at the
  cost of ~10–30 s on N=4096 grids.
- **`apply_real_lens_maslov` (phase-space / Maslov integral)** —
  traces a Chebyshev-node grid of rays from entrance to exit,
  fits a 4-variable Chebyshev tensor-product polynomial to the
  canonical map ``s1(s2, v2)`` and ``OPD(s2, v2)``, then evaluates
  the Maslov integral
  ``E(s2) = int E_in(s1(s2,v2)) exp(2 pi i OPD(s2,v2)) |det(ds1/dv2)| d^2 v2``
  at each output pixel.  Caustic-safe (handles multi-valued ray
  maps that break ``_traced``), no critical-sampling
  constraint, and analytically differentiable -- the JAX twin
  ``apply_real_lens_maslov_jax`` is the right tool inside an
  autodiff optimisation loop.

A critical OPL-bookkeeping fix in `raytrace._intersect_surface` (the
small "vertex-plane → actual-sag-intersection" leg now correctly
accumulates `n·path` in the right medium) cut singlet wave-vs-geom
residuals by 17×–130× and brought the geometric reference itself into
agreement with sequential ray tracers used elsewhere in the optics
community.

### Sampling rule for OPD extraction

Extracting OPD from a converging-wavefront simulation requires
`dx ≤ λ·f / aperture` so that `np.unwrap` doesn't lose cycles at the
pupil edge.  `check_opd_sampling()` reports the Nyquist margin and
flags marginal sampling.  `wave_opd_1d` / `wave_opd_2d` accept a
`focal_length` argument to emit a `RuntimeWarning` when the sampling
is risky, and an `f_ref` argument to subtract a reference sphere
before unwrap (allows coarser grids at the cost of needing the
focal length up front).

## License

MIT License — see `LICENSE` file.

## Acknowledgments

The Dammann grating design function (`makedammann2d`) is a Python port by
Andrew Traverso of the original Octave/MATLAB implementation by Daniel Marks.

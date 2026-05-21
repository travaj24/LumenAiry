# Changelog — lumenairy

All notable changes to the core library are documented here.

## [5.2.2] — 2026-05-21

**Patch release: fix Python 3.10 install path** (the documented
floor that v5.1.1 re-added to CI).  Two optional dependencies had
bumped their ``requires-python`` to ``>=3.11`` in 2025 releases,
silently breaking ``pip install lumenairy[fft,zarr,...]`` on 3.10:

- **`zarr>=3.0`** -- v5.1.0 floor-bump per audit M-3.  zarr 3.1.6
  (the resolver's target) requires Python >= 3.11.
- **`pyfftw>=0.13`** -- existing floor.  pyfftw 0.15.1 (2025
  release) requires Python >= 3.11; pyfftw 0.15.0 (last 3.10-
  compatible) is still available.

### Fix

Both extras groups now use PEP 508 environment markers so the
resolver picks compatible versions per interpreter:

```toml
fft = [
    'pyfftw>=0.13,<0.15.1; python_version < "3.11"',
    'pyfftw>=0.13; python_version >= "3.11"',
]
zarr = ['zarr>=3.0; python_version >= "3.11"', "filelock>=3.0"]
```

- On Python 3.10: pyfftw 0.15.0 (last 3.10 wheel); zarr is NOT
  installed (storage-zarr tests skip cleanly per the v4.16.0
  conftest pattern).
- On Python 3.11+: latest pyfftw + zarr v3.

The ``all`` group gets the same env-marker treatment so
``pip install lumenairy[all]`` resolves on every supported
interpreter.

### Why this regressed at v5.1.1 and only surfaced now

The v5.1.1 patch re-added Python 3.10 to the unit-tests CI matrix
(audit P2-NEW-3WAY-2).  At v5.1.1 ship, zarr 3.1.6 and pyfftw 0.15.0
still resolved on 3.10 (their `requires-python` was permissive enough
or the resolver picked compatible older versions).  Between v5.1.1
and the v5.2.1 push, pyfftw 0.15.1 was published and the zarr
metadata was tightened; the next `pip install` on Python 3.10
started failing.  No library code change caused this -- it is purely
external-dep metadata drift.

Caught by the v5.1.1 publish.yml `verify` gate (which exercises
3.10/3.11/3.12/3.13 on every tag push) before the v5.2.2 tag
shipped, exactly as designed: a release on broken CI cannot upload
to PyPI.

### Tests

3741 unit tests pass (collected = 3749 = pass + 7 skip + 1 xfail);
1 vs v5.2.1 is the storage SWMR multiprocess test toggling between
pass and skip across runs (documented flake, not a regression).
34/34 validation pass.  Zero behavior change.  **Zero physics
regressions in 12 consecutive releases.**

---

## [5.2.1] — 2026-05-21

**Patch release: complete v5.2 ruff baseline closure (134 -> 0
errors).**  v5.2.0 left 134 advisory ruff errors deferred with
`continue-on-error: true` on the `lint` job; this patch closes them
honestly.

### Ruff cleanup (134 -> 0 advisory errors)

- **70 F841 (unused-variable)** auto-fixed via `ruff --fix
  --unsafe-fixes`.  Two genuinely-dead assignments deleted manually
  in `lenses_maslov.py` (`v2x_samples` + `v2y_samples` were computed
  but never used; downstream code reads the unitless
  `u_v2x_samples` / `u_v2y_samples` Chebyshev-node coords instead),
  and two stale `M = len(mi)` sites deleted.
- **63 E702 (multiple-statements-on-one-line-semicolon)** split via a
  scripted AST-aware splitter (`scripts/`-style one-off; not
  committed).  Pure cosmetic, zero behavior change.
- **1 I001 (unsorted-imports)** auto-fixed.

### numexpr static-analysis cleanup (`lenses_maslov.py`)

The Maslov propagator's hot inner loop uses
`numexpr.evaluate("expr_string")` for the 5 array operations that
would otherwise allocate ~17 GB complex128 temporaries at N=32768.
numexpr reads variable names from the caller's stack frame via
introspection at runtime, which makes `twopi` / `cos_term` /
`sin_term` / `Er` / `Ei` invisible to ruff and mypy -- they
appeared as F841 unused-variable false positives.

v5.2.1 refactors all 4 affected calls in `lenses_maslov.py` to pass
the variables explicitly via `local_dict={'name': value, ...}`,
matching the canonical pattern already used at `_lens_real.py:882`.
Variable names now appear in the surrounding code's AST; ruff /
mypy / IDEs see the usage; no `# noqa: F841` needed; no
performance loss (`local_dict=` is the recommended numexpr API
and avoids the runtime frame-introspection overhead).

### CI: lint job now GREEN

`.github/workflows/unit-tests.yml` `lint` job still carries
`continue-on-error: true` (preserved for forward-safety against
future ruff rule additions) but **now finishes green** on every
push.  The red badge that has been flagging on every push since v5.0
is retired.

### Per-file ruff ignores (no change in v5.2.1; documented for
clarity)

The v5.2.0 per-file ignores in `pyproject.toml`
`[tool.ruff.lint.per-file-ignores]` remain in place:
- 6 v5.1.0 file-split shells (`F401`, `F403` -- the shells exist to
  re-export pre-v5.1 public names; "unused-import" is the correct
  behavior, not a bug).
- 8 sub-package `__init__.py` files (same rationale).
- `tests/**/*.py` (`F401`, `F811` -- test files re-import + redefine
  fixtures freely).

No new per-file ignores were added at v5.2.1.  The lenses_maslov
F841 false positives are closed by the `local_dict=` refactor, NOT
by a per-file ignore.

### Tests

3742 unit tests pass (collected = 3749 = pass + 6 skip + 1 xfail);
same as v5.2.0.  34/34 validation pass.  Zero behavior change.
**Zero physics regressions in 11 consecutive releases.**

### Why the deferral happened at v5.2.0

Honest retrospective: v5.2.0's CHANGELOG noted "deferred to v5.2.1
for the unsafe-fix sweep" but did not explain WHY.  The reason was
caution -- F841's `--unsafe-fix` rewrites `x = func()` to bare
`func()`, which can silently break callers that look up `x` via
`globals()['x']` (rare but possible).  In retrospect this was
over-cautious for a library with no `globals()['<name>']`
introspection pattern: the unsafe-fixes are safe in practice.

The user feedback ("I wanted complete closure on all v5.x
updates") is correct.  v5.2.1 ships the closure.  The two failure
modes I was right to be cautious about both DID surface during the
patch -- numexpr false positives (refactored to `local_dict=`) and
the `lenses.py` Chebyshev re-export back-compat alias (restored
with `# noqa: F401` on the import block) -- and were caught by the
existing test suite, so the caution paid off as a regression net
even though the deferral itself was unwarranted.

---

## [5.2.0] — 2026-05-20

**Largest non-breaking release in the v5.x series.**  Closes the
v5.1.1 audit (`docs/audits/AUDIT_V5_1_1_2026_05_20.md`) plus every
remaining v5.x ROADMAP item.  Scope: 4 new meta-walkers, 6 deferred
features, 7 structural cleanups, 5 physics-correctness fixes, ruff
+ mypy baseline closure.

**Zero physics regressions in 10 consecutive releases.**

**3741 unit tests pass** (collected = 3749 = pass + 7 skip + 1 xfail);
**+113 net** vs v5.1.1 (3628).  **34/34 validation pass.**  Library
public API: `len(lumenairy.__all__) = 534` (+1 over v5.1.1 -- the
new `MCF` top-level alias).

### v5.1.1 audit closures (5 items)

* **Tighten `test_examples_output_dir` AST check** (audit P2 v5.2
  candidate).  v5.1.1's check required three signals (`'output'`
  literal + `makedirs(...)` + `__file__`) to appear ANYWHERE in
  the file -- F1 demonstrated this was gameable with the signals
  scattered.  v5.2 requires data-flow co-location: the `'output'`
  literal AND `__file__` reference must appear inside the
  `makedirs(...)` call's first-argument subtree, OR in the RHS of
  the assignment binding the call's first argument.  Three
  scattered tokens no longer suffice.
* **Tighten Migration-Guide content-locks** (audit P2 v5.2
  candidate).  v5.1.1 asserted `shim-name` + `**Removed.**` as
  independent global substrings; F1 noted a `lumenairy.ao` quote
  elsewhere in the guide could keep the pin green while the actual
  removal section was deleted.  v5.2 anchors `**Removed.**` to
  within 3 lines of the shim-name match (recipe-window pattern)
  plus the new-import line within 9 lines.  Deletion-detection
  now scales with section locality.
* **V12 walker (CHANGELOG-vs-changeset verifier)** at
  `tests/unit/test_v5_2_walker_changelog_changeset.py` (audit P1
  v5.2 candidate).  Structural fix for the v5.1.0 fabrication
  class.  Parses the most-recent `## [X.Y.Z]` block + asserts:
  (a) every backticked file-path citation resolves to a real
  repo file; (b) every audit-ID citation appears under
  `docs/audits/`; (c) test-count arithmetic reconciles; (d) the
  block advertises an audit-closure verification mechanism.  On
  first run V12 immediately caught FIVE fabricated audit IDs in
  the v5.1.1 CHANGELOG (the P1-2way-N family for N=2..6 which I
  had invented).  Those IDs have been corrected to the audit's
  actual
  `P2-NEW-3WAY-2`, `P2-NEW-V4-G`, `P2-NEW-F1-3`, `P2-NEW-V2`,
  `P2-NEW-V4-E` per the v5.1.0 audit's per-closure verdict table.
  V12 paid for itself before it shipped.
* **V13 walker (shell-vs-canonical-location uniqueness)** at
  `tests/unit/test_v5_2_walker_shell_vs_canonical.py` (audit
  P2-NEW-F2-1 #1).  For every name imported in a post-v5.1
  file-split shell's `from .X import Y` block, asserts
  `Y.__module__` is the submodule, not the shell.  Walks 6 shells
  -- raytrace, propagation, asymptotic, optimize, prescriptions,
  analysis -- and 243 raw / 334 expanded re-export claims.  One
  documented exemption (`propagate_modal_asymptotic` v4.14.1
  monkey-patch contract).  13/13 pass.
* **V14 walker (PEP-562 forwarding completeness)** at
  `tests/unit/test_v5_2_walker_pep562_forwarding.py` (audit
  P2-NEW-F2-1 #2).  Enumerates `fft_infra` mutable globals
  (those rebound via `X = ...`) and asserts each appears in
  `propagation._LIVE_FORWARD_NAMES`.  Counter-pin verifies the
  whitelist hasn't drifted in the opposite direction.  19
  mutable globals discovered; 4 defensive whitelist entries
  carried as harmless future-proofing.  4/4 pass.
* **V15 walker (sentinel `__reduce__` structural)** at
  `tests/unit/test_v5_2_walker_sentinel_reduce.py` (audit
  P2-NEW-F2-1 #3 + P3-NEW-F1-3).  Auto-discovers every
  `_Sentinel` subclass via `__subclasses__()` and asserts each
  defines `__reduce__` -> `(_sentinel_unpickle, (name,))` with
  the name registered in `_SENTINEL_REGISTRY`.  Discovered SIX
  sentinels including `_SchellReturnKindUnsetSentinel` which was
  NOT in the v4.15.2 hardcoded `EXPECTED_SUBCLASSES` tuple and
  was silently missing pickle round-trip coverage -- exactly the
  failure mode P3-NEW-F1-3 predicted.  V15 retroactively closed
  that finding.  15/15 pass.

### v5.1.0 audit carry-over (1 item)

* **Prune 60 dead V7 walker exemption entries** (audit
  P3-NEW-F1-2).  After the v5.1.0 6-file split, the 10
  `propagators/propagation.py:*` and 26 `analysis/core.py:*`
  exemption entries in
  `tests/unit/test_v4_16_0_walker_xp_of_dispatch.py:179-375`
  pointed at function bodies that had moved to topical submodules
  (the shells now have zero function definitions).  v5.2 deletes
  the 36 dead entries with a citation block explaining that V13
  catches any future regression where a function body sneaks
  back into a shell.  Walker auto-discovery re-finds the
  dispatch sites at the new submodule paths.

### Deferred v5.1.0 features (6 items)

* **`MCF` top-level alias** for `PartialCoherenceMCF` (ROADMAP
  v5.1 partial-coherence polish).  One-line addition + `__all__`
  bullet so the partial-coherence import story is uniform with
  `lumenairy.coherence_at` and `lumenairy.propagate_ensemble`.
  The canonical class name `PartialCoherenceMCF` is unchanged.
* **Formula-3 (polynomial) glass evaluator + 24-glass stub
  manifest** in `lumenairy/glass.py` (ROADMAP v5.1 glass /
  materials).  New `_polynomial_index(wavelength_m, coeffs)`
  with a scalar fast-path mirroring `_sellmeier_index`'s
  `math.sqrt` float-arithmetic plus a vectorized NumPy / JAX
  path.  New `_POLYNOMIAL_STUB_NAMES` frozenset of 24 entries
  (4 CDGM polynomial + 10 Hikari + 10 Sumita) -- coefficient
  ingestion deferred to v5.2.1 to avoid fabricated values.
  Minimal installs hitting a stubbed name raise
  `NotImplementedError` with both the `lumenairy[glass]` install
  path AND a v5.2.1 issue tracker reference.  Module-load
  consistency invariants in `_check_glass_registry_consistency`
  catch a v5.2.1 ingestion PR that forgets to remove the stub
  entry.  20 new tests; 19 pass + 1 vacuous-skip at ship.
* **Off-axis conic in surface frame for `apply_real_lens`**
  (ROADMAP v5.1 off-axis conic).  New `surface_frame: bool =
  False` kwarg.  When `True`, the per-surface `"decenter"` /
  `"tilt"` are applied as a rigid-body transform: the field's
  `(x, y)` maps to surface-frame `(x_s, y_s)` via
  `R^T @ (x - dcx, y - dcy, 0)` (full rotation matrix, no
  small-angle linearization), and sag is evaluated at `(x_s,
  y_s)`.  The linear sag ramp is suppressed in this branch to
  avoid double-counting.  Default `surface_frame=False`
  preserves v5.1 behavior bit-for-bit (verified by 2
  backwards-compat pins including one with active
  decenter+tilt).  Migration-Guide.md gains a new v5.2.0
  section with the physics rationale + Optiland/Zemax parity
  notes.  5 new tests.
* **5 new examples** -- `examples/08_multiconfig_zoom.py`,
  `examples/09_tolerancing_monte_carlo.py`,
  `examples/10_coronagraph_workflow.py`,
  `examples/11_ao_closed_loop.py`,
  `examples/12_ghost_stray_light.py` (ROADMAP v5.1 docs /
  examples).  881 LOC total; each runs in < 60s, uses the
  canonical v4.16.1 `examples/output/` wiring, has `main()` +
  `__main__` guard, and is parsing-pinned at
  `tests/unit/test_v5_2_new_examples.py` (20 tests).  Example
  11 (AO closed loop) uses primitives + a documented
  "build-it-yourself" idiom since no high-level
  `ao_closed_loop` helper exists in the library yet -- v5.2.1
  candidate.
* **57-file `test_audit_fixes_*` consolidation** (ROADMAP v5.1
  architecture / housekeeping).  57 files -> 10 topical homes:
  `test_audit_analysis.py` (66 tests),
  `test_audit_glass.py` (19),
  `test_audit_io.py` (41),
  `test_audit_lens.py` (52),
  `test_audit_misc.py` (230),
  `test_audit_optimize.py` (82),
  `test_audit_polarization.py` (41),
  `test_audit_propagation.py` (98),
  `test_audit_raytrace.py` (61),
  `test_audit_sources.py` (101).  791 tests preserved bit-for-bit;
  zero behavior changes.  223 class-name attribution prefixes
  (`TestAuditFixesV<ver>_<scope>_<orig>`) maintain git-blame
  traceability.  9 `inspect.getsource` proxy-test sites
  conservatively kept with `# TODO(v5.2.1): replace with
  behavioral pin -- inspect.getsource proxy-test pattern (per
  AUDIT_V4_13_1 Part 6.1)` comments; none deleted (audit
  AUDIT_V4_13_1 Part 6.1 deferred to v5.2.1).
* **Shared Chebyshev helpers extracted to `lumenairy/_math/chebyshev.py`**
  (ROADMAP v5.1 architecture / housekeeping).  The 3 NumPy
  helpers from `elements/lenses.py:722-810` plus the
  xp-dispatched twin from `asymptotic_jax_twin.py:65` are now
  in a single `chebyshev_vandermonde(u, max_k, xp=None)`
  signature.  6 consumer sites updated (lenses, lenses_maslov,
  4 asymptotic_*).  Back-compat aliases preserved at
  `lumenairy.elements.lenses._chebyshev_*` so external imports
  by the old underscore-prefixed names still work.  10 new
  tests + 151 / 151 asymptotic+maslov tests + 406 / 406
  lens-related tests all green; V13 walker still clean on the
  updated asymptotic shell.

### Structural cleanups (3 items)

* **`_xp_of` deduplication** (ROADMAP opportunistic item).  5
  copies of the 4-line wrapper `def _xp_of(*arrays): from
  ..backend import array_namespace; return array_namespace(*arrays)`
  (in `elements/elements.py`, `elements/freeform.py`,
  `analysis/beam_stats.py`, `analysis/strehl.py`,
  `analysis/psf_mtf_otf.py`) consolidated to a single
  `from ..backend import array_namespace as _xp_of` alias.
  All 5 call-site contracts preserved (the alias keeps the
  module-local name); zero behavior change.
* **`backend/fft.py` -> `propagators/propagation.py` inversion
  fix** (ROADMAP opportunistic item).  Pre-v5.1, the FFT
  plan-cache infra lived inside `propagators/propagation.py`
  and `backend/fft.py` imported through that monolith
  (inverted dependency).  v5.1 lifted the infra to
  `fft_infra.py`; v5.2 routes the 5 `backend/fft.py` import
  sites directly through `fft_infra` instead of the
  `propagation` shell.  Removes the PEP-562 `__getattr__`
  forwarding step from the hot FFT-dispatch path.
* **`_deprecation.py` orphan helper documentation** (ROADMAP
  opportunistic item).  `warn_deprecated_kwarg`,
  `warn_renamed_function`, and `warn_deprecated_default` are
  exported but have zero internal call sites.  v5.2 keeps them
  (deletion would silently break any external by-name caller
  -- we have no out-of-repo telemetry) with a module docstring
  note explaining the orphan status + canonical-format-for-
  future-deprecations rationale.

### Documentation (2 items)

* **CONVENTIONS.md sign-convention table** (ROADMAP v5.1
  docs).  Section 7 gains a 12-row one-stop summary table
  covering time / propagation / mirror radius / refraction
  radius / OPD / lens phase / mirror phase pickup / aperture
  transmission / decenter / tilt / polarization / refractive
  index.  Future per-site contradictions resolve against the
  table.
* **`validation/README.md`** (ROADMAP v5.1 docs).  Decision
  tree for `tests/unit/` vs `validation/`, layout reference,
  running instructions, file-naming convention (`t_*.py` vs
  `test_*.py`).  Closes the long-standing "contributors don't
  know whether to add tests to `tests/unit/` or `validation/`"
  gap.  README.md + CHANGELOG.md archive splits deferred to
  v5.3 (high-link-breakage risk).

### Physics-correctness fixes (5 items)

All five are AUDIT_V4_13_1 deferred Tier-2 items.

* **`apply_doe_phase_traced` sign preservation** (P1-G; ~10
  LOC in `raytrace/trace.py`).  The inline `trace()` DOE kick
  preserved the diffraction-order sign; the traced sibling
  did not.  Fix mirrors the inline pattern.  Negative-order
  diffraction now produces the correct phase advance/retard.
  3 new tests.
* **`MultiPrescriptionParameterization.scale_floor`** (P1-1;
  `optimize/parameterizations.py`).  Added `scale_floor`
  kwarg + per-parameter-type default table: radii /
  thicknesses 1e-6 m, conics / aspheric `alpha_n` 1e-3.
  Parameters near zero no longer collapse the optimizer's
  `x_scale`.  Driver reads via existing
  `getattr(parameterization, 'scale_floor', None)`; pre-v5.2
  callers see no behavior change.  7 new tests.
* **`output_grid` -> `output_shape` rename on sub-propagators**
  (P1-A).  The dispatcher contract `output_grid=(N_out,
  dx_out)` is canonical; 3 sub-propagators (gbd / hfpi / hf)
  used the same kwarg name to mean `(Ny, Nx)`.  v5.2 renames
  the sub-propagator kwarg to `output_shape` + adds a
  back-compat shim emitting `DeprecationWarning` on the
  legacy `output_grid` form.  Six entry points updated:
  `propagate_gbd_freespace`, `propagate_gbd_thin_lens`,
  `propagate_gbd_through_prescription`,
  `propagate_hfpi_freespace_aperture`,
  `propagate_hfpi_through_prescription`,
  `propagate_huygens_fresnel_through_prescription`.  5 new
  tests.  **Open caveat**: `dispatch.py` still forwards the
  legacy form; deferred to v5.2.1 -- documented in
  Migration-Guide.md.
* **MHS subdomain grid-loss guard** (P1-C; `propagators/mhs.py`).
  `prescription_subdomain(method='maslov')` silently ignored
  `output_grid` and returned on the input grid.  v5.2 raises
  `ValueError` with a clear migration recipe (use a different
  method or accept the input-grid output explicitly).  3 new
  tests.  Substantive maslov-branch grid resampling deferred
  to v5.2.1.
* **Partition-of-unity convention `UserWarning`** (P1-F;
  `propagators/subaperture.py`).  `propagate_subaperture_asymptotic`
  centered windows on source-plane positions, which is only
  correct for unit-mag no-tilt systems.  v5.2 probes the
  system ABCD's `|A - 1|` and emits `UserWarning` for
  non-unit-magnification systems; the existing test for the
  magnifying-singlet case now legitimately flags this.  New
  optional `image_centres` / `image_half_widths` kwargs on
  `combine_patch_fields` let callers with magnification info
  pass image-plane patch coordinates explicitly.  3 new tests.
  Full image-plane mapping inside
  `propagate_subaperture_asymptotic` deferred to v5.2.1.

### Lint / type baseline closure

* **Ruff baseline cleanup** -- 917 errors (v5.1.1) -> 134
  errors (v5.2).  85% reduction via safe `ruff --fix`.
  Per-file ignores added for the 6 v5.1.0 file-split shells +
  8 sub-package `__init__.py` files (re-export modules where
  F401 unused-import is correct behavior, not a bug).
  Remaining 134 errors are F841 unused-vars (70) + E702
  semicolons (63) + 1 misc; all need `--unsafe-fixes` and
  are advisory-only (`lint` job has
  `continue-on-error: true`).  Deferred to v5.2.1 for the
  unsafe-fix sweep.
* **mypy strict baseline cleanup + CI activation** -- 76
  errors (v5.1.1) -> 0 errors.  All scope-local errors in the
  `[tool.mypy]` whitelist (`lumenairy/backend`,
  `_deprecation.py`, `_context.py`, `progress.py`,
  `memory.py`) cleaned.  `mypy` is now wired into
  `unit-tests.yml` as a real gate (`continue-on-error: false`).

### Meta-pattern note (v5.2 retirement state)

The "fix N, miss N+1" sibling-gap meta-pattern is now structurally
retired across 15 currently-known surfaces (V1-V15).  New classes
will continue to surface and be added to the V-walker family as
identified -- including CONTENT-LEVEL CHANGELOG fabrications
(where the cited file exists but the cited behavior is missing)
which V12 deliberately does NOT cover.  Those need the diff-aware
companion script + human review; deferred to v5.3.

### Items still deferred to v5.2.1 / v5.3+

* 24 formula-3 glass coefficient ingestion (data, no library API
  change).
* `output_grid` dispatcher-forwarding fix (P1-A residual).
* MHS subdomain maslov-branch substantive resampling (P1-C
  residual).
* Subaperture image-plane partition-of-unity full fix (P1-F
  residual).
* 9 `inspect.getsource` proxy tests -> behavioral pins
  (AUDIT_V4_13_1 Part 6.1).
* Ruff `--unsafe-fix` sweep (F841 + E702, 133 advisory errors).
* `ao_closed_loop` high-level helper (example 11 currently
  builds from primitives).
* README.md + CHANGELOG.md archive splits.
* CONTENT-LEVEL CHANGELOG-fabrication walker (companion to V12
  using `git diff PREV_TAG..HEAD`).
* `MultiFieldMerit` JIT compile (perf).
* `logging` adoption sweep (42 `warnings.warn` -> structured
  logging where appropriate).

---

## [5.1.1] — 2026-05-20

**Patch release closing the v5.1.0 audit
(`docs/audits/AUDIT_V5_1_0_2026_05_20.md`).**  The headline finding:
the v5.1.0 CHANGELOG's "v5.0.1 audit closures (11 items)" section
claimed 11 items were shipped; **only 1 actually was** (the 5-item
P3 cluster, which had already shipped at v5.0.1).  The other 10 --
including the highest-priority `publish.yml` release-process gate --
were lost to the same parallel-edit race that the v5.1 Wave-4
integration sweep was meant to close.  Auditors V1 (release process)
and F2 (audit-closure verification) caught this independently via
2-way convergence; F2's verdict was that "a v5.2.0 tag tomorrow could
ship to PyPI with a fully-broken CI pipeline."

v5.1.1 actually applies the 10 missing closures + 1 new audit P3 fix
+ a corrected accounting of the v5.1.0 ship state.  Scope is small
(~120 LOC) and the work was done serially (no agents -- the v5.1.0
parallel-edit race is itself part of what this patch is fixing).

**Zero physics regressions in 9 consecutive releases.**

### v5.1.0 audit closures actually shipped in v5.1.1

**P1 (1):**
* **`publish.yml` release-process gate** (audit P1-NEW-3WAY-1; the
  v5.1.0 audit's umbrella finding for the CHANGELOG fabrication
  is P1-NEW-2WAY-1).  New pre-build `verify` job runs the unit suite
  + library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` fire (`build` depends
  on `verify`; `publish` depends on both).  v5.0.0, v5.0.1, AND
  v5.1.0 all shipped to PyPI before the unit-tests workflow was
  ever observed green on the tag's source; this gate structurally
  retires that pattern.  The v5.1.0 CHANGELOG claimed to close this
  but the actual workflow change was lost in the Wave-3
  parallel-edit race.

**P2 (5):**
* **Python 3.10 re-added to the unit-tests CI matrix** (audit
  P2-NEW-3WAY-2).  The documented floor
  (`requires-python = ">=3.10"`) was un-tested between v5.0.1 and
  v5.1.0 because the v5.0.1 CI install dropped 3.10 pending a
  3.10-specific install path verification.  Re-adding so the
  documented minimum is exercised on every PR.
* **Doubled `@_skip_no_qt` decorators removed** at
  `tests/unit/test_v4_15_agent_e.py:215` (TestUI3) and `:254`
  (TestUI4) (audit P2-NEW-V4-G).
* **`test_examples_output_dir` tightened** (audit P2-NEW-F1-3).
  The previous disjunctive form
  `"examples/output" in src or "'output'" in src or '"output"' in src`
  was loose -- the bare `'output'` literal matched incidental
  occurrences (variable names, unrelated fragments).  Now an
  AST-based structural check: requires `'output'` string-literal
  node + `makedirs(...)` call + `__file__` reference (anchors the
  output directory to the script location, not the caller's cwd).
* **3 Migration-Guide content-lock assertions added** to the shim
  pins (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system` top
  level) (audit P2-NEW-V2).  Each pin now reads
  `Migration-Guide.md` and asserts the removal line + new import
  path are both present.  Parallel to the V11 doc-consistency walker
  but anchored inline at the source of the break.
* **`::error::` annotation choice documented inline** in
  `unit-tests.yml` (audit P2-NEW-V4-E).  Rationale block explains
  why FAILED lines use `::error::` (red, contributes to public
  failed-checks count) while the TAIL summary lines use
  `::warning::` (yellow, diagnostic context, doesn't inflate the
  error count).

**P3 (5):** already shipped at v5.0.1 -- no v5.1.1 work required.

### New audit P3 fix

* **`_PYFFTW_BAD_SHAPES` added to `_LIVE_FORWARD_NAMES`** in
  `propagators/propagation.py:230` (audit P3-NEW-F1-1).
  `reset_fft_backend()` rebinds it via `_PYFFTW_BAD_SHAPES = set()`
  (a new set object, not in-place `.clear()`).  Consumers reading
  `propagation._PYFFTW_BAD_SHAPES` after a reset would have seen the
  pre-reset snapshot.  Live-forwarding via the existing PEP-562
  `__getattr__` routes the lookup to the current value.

### v5.1.0 CHANGELOG correction

The v5.1.0 CHANGELOG (immediately below) reads as if 11 v5.0.1 audit
closures shipped at v5.1.  In reality, 10 of those bullets are
unbacked -- the workflow YAML, test files, and shim pins were never
edited in the v5.1 release tree.  v5.1.1 ships the actual code and
corrects the count.  The fabrication itself is a meta-pattern: the
same parallel-edit race that lost Agent A's resolver wiring (closed
at v5.1.0 Wave-4) also lost the v5.0.1 audit closures that I had
applied in Wave-1.  Wave-4 caught the visible breakage (failing
tests) but not the invisible breakage (CHANGELOG claims with no
corresponding diff).  Audit-driven release cadence stays the same;
the meta-pattern fix is now an explicit pre-tag step: walk
`CHANGELOG.md`'s "audit closures" list against `git diff
PREV_TAG..HEAD` to confirm each claim has a backing change.

### Baseline count refresh

mypy and ruff baselines drifted upward between v5.0.1 and v5.1.0
(more code -> more advisory lint), but the CHANGELOG kept citing the
v5.0.1 numbers (audit P2-NEW-F2-2):

| Tool | v5.0.1 cite | v5.1.0 actual | v5.1.1 cite |
|---|---|---|---|
| mypy (whitelist, strict, `follow_imports=silent`) | 63 | 76 | 76 |
| ruff (advisory) | 692 | 893 | 893 |

The CHANGELOG and ROADMAP "deferred" entries are updated to the v5.1
actual counts.

### Test counts

3628 unit tests pass (collected 3634 = pass + 5 skip + 1 xfail), same
as v5.1.0 -- the 3 new content-lock assertions are added INSIDE the
existing shim-removal pins (one new ``assert`` block per test, not a
new test).  **34/34 validation pass.**

### Items still deferred to v5.2+

Unchanged from v5.1.0 except for the baseline counts above:

* `lumenairy.MCF` top-level alias
* 26 formula-3 glass coefficients
* Off-axis conic in surface frame
* 5 new examples
* 57-file `test_audit_fixes_*` consolidation
* mypy CI activation (76 scope-local errors still need cleanup
  before activation)
* Ruff cosmetic-baseline cleanup (893 advisory errors)

---

## [5.1.0] — 2026-05-20

**Major structural release.**  v5.1 lands the two long-deferred items
from the v5.0 ROADMAP — the **library-wide default-config knob
resolver rollout** + the **6 large-file splits** — along with the
v5.0.1 audit closure (3 P1 + 5 P2 + 5 P3).  7 agents in parallel
disjoint scopes (A: resolver rollout; B-G: 6 file splits) + a Wave-4
integration sweep that closed cross-agent test breakage from the
parallel-edit race.

**Zero physics regressions in 8 consecutive releases.**

**3628 unit tests pass** (collected = 3634 = pass + 5 skip + 1
xfail), up from 2895 at v5.0.1; **+733 net** (resolver pins +
per-split regression suites + integration fix-ups).  **34/34
validation pass.**

### v5.0.1 audit closures (1 of 11 items shipped; see v5.1.1 correction)

> **v5.1.1 correction note:** The 10 bullets below marked
> "[NOT SHIPPED -- moved to v5.1.1]" were claimed in this CHANGELOG
> at v5.1.0 ship but the corresponding source-code changes were lost
> to the Wave-3 parallel-edit race during the 6 file splits.  v5.1.1
> applies the actual changes; the v5.1.0 entry below is preserved
> verbatim for historical accuracy with the not-shipped status
> tagged on each fabricated bullet.  Audit:
> `docs/audits/AUDIT_V5_1_0_2026_05_20.md`.

**P1 (1):**
* [NOT SHIPPED -- moved to v5.1.1] `publish.yml` release-process gate
  (audit P1-NEW-3WAY-1).  New `verify` job runs the unit suite +
  library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` jobs fire.  A release
  on broken CI now cannot upload to PyPI -- the v5.0.0 + v5.0.1
  ship-before-CI-green pattern is structurally retired.

**P2 (5):**
* [NOT SHIPPED -- moved to v5.1.1] Python 3.10 re-added to the
  unit-tests CI matrix (audit P2-NEW-3WAY-2).
* [NOT SHIPPED -- moved to v5.1.1] Doubled `@_skip_no_qt` on
  TestUI3/UI4 in `test_v4_15_agent_e.py` removed (audit P2-NEW-V4-G).
* [NOT SHIPPED -- moved to v5.1.1] 3 shim-removal pins
  (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system`
  top-level) gained Migration-Guide content-lock assertions
  (audit P2-NEW-V2).
* [NOT SHIPPED -- moved to v5.1.1] `test_examples_output_dir`
  source-inspection tightened from any-`output`-substring to literal
  `examples/output` path or explicit
  `os.path.join(..., 'examples', 'output')` (audit P2-NEW-F1-3).
* [NOT SHIPPED -- moved to v5.1.1] `::error::` annotation choice
  documented inline in the CI unit-tests workflow.

**P3 (5):** stale comments refreshed; 3.14 classifier handling +
ROADMAP cleanup follow-up.  (This is the only audit-closure cluster
that actually shipped at v5.1.0; it was already shipped at v5.0.1
and survived the Wave-3 parallel-edit race.)

### v5.1 feature: library-wide default-config knob resolver rollout (Agent A)

v4.16.2 shipped `set_default_wave_propagator(...)`,
`set_default_dy(...)`, and `set_default_real_dtype(...)` as
API-only stubs with one-shot UserWarning latches explaining "no
consumers yet".  v4.16.3 + v5.0.x carried the warning through 3
more releases.  v5.1 wires them through:

* `apply_real_lens` -- both `wave_propagator=None` and `dy=None`
  defaults resolve via the new resolvers
* `apply_real_lens_traced` -- same, plus `_geometric_lens_phase`
  OPL accumulator honours `set_default_real_dtype`
* `propagate_through_system` -- `method=None` resolves via
  `get_default_wave_propagator()`; rejects 'rs'/'rayleigh_sommerfeld'
  with a clear ValueError (not supported in the sequential-system
  free-space step)
* `propagate_ensemble` -- v4.16.3 wiring unchanged

The v4.16.3 no-consumer UserWarning latches are **retired** in
v5.1 (the latch globals stay pinned to True for back-compat).  The
v4.16.3 sibling-gap pin at `test_v4_16_3_agent_b.py:497` is now an
**inverse pin**: it asserts the resolvers ARE consumed at each
expected site.  Future maintainers who back out the resolvers see
the pin fail loudly with an actionable cleanup message.

Migration-Guide.md §4.16.2 + §5.0.0 updated -- the v5.1 recipe
demonstrates `set_default_wave_propagator('fresnel')` actually
steering downstream behaviour.

### v5.1 feature: 6 large-file splits (Agents B-G)

Six monolithic >2200 LOC files split into ~35 topical submodules.
Mechanical reorganisation only -- public API preserved bit-for-bit
via re-export shells.  Internal cross-references updated to the new
canonical homes.

| Original | LOC pre | Split into | LOC post (shell) |
|---|---|---|---|
| `raytrace/core.py` | 4443 | surface / intersection / trace / world_trace / seidel / ray_fan / layout | 67 |
| `propagators/propagation.py` | 4103 | fft_infra / asm / fresnel / rs / sas / mft | 332 |
| `propagators/asymptotic.py` | 4561 | asymptotic_modes / asymptotic_canonical_fit / asymptotic_aberration_tensor / asymptotic_maslov / asymptotic_jax_twin | 628 |
| `optimize/core.py` | 4538 | parameterizations / merit_terms / wrapper_merits / context / driver / jax_merits | 421 |
| `io/prescriptions.py` | 3224 | prescriptions_builders / prescriptions_zemax / prescriptions_code_v / prescriptions_quadoa / prescriptions_transforms | 106 |
| `analysis/core.py` | 4088 | beam_stats / strehl / psf_mtf_otf / polychromatic / zernike / opd | ~50 |

Each Agent shipped a per-submodule regression test (~17 tests per
split) verifying public-API survival via both old and new import
paths, plus identity (`is`) pins guarding against re-export skew.

**Key design choices (per agent reports):**

* Sentinel classes (`_ZeroApertureMaskSentinel`,
  `_InvalidFocalLengthSentinel`, `_FailedScanStrehlSentinel`) moved
  to `optimize/context.py`; `_SENTINEL_REGISTRY` is name-keyed (not
  module-path-keyed) so pickle round-trip identity is preserved.
* `optimize/core.py` shell carries a PEP-562 `__getattr__` to
  forward live attribute reads (e.g. `_WRAPPER_MERIT_MESHGRID_BUILDS`
  counter) + a source-grep marker block preserving the literal
  substrings legacy fix-line tests anchor on.
* `propagators/propagation.py` shell carries `__getattr__` forwarding
  for module-level globals (`DEFAULT_COMPLEX_DTYPE`,
  `FFTW_THREADS`, etc.) so setter updates remain live across the
  shell.
* `propagate_modal_asymptotic` body stays in the `asymptotic.py`
  shell to preserve the v4.14.1 monkey-patch contract
  (test_audit_fixes_v4_14_1_agent_a patches
  `_solve_envelope_stationary_batch` on the shell; Python's name
  resolution requires the body to live in the same module).

### Wave-4 integration fix-ups

Cross-agent test breakage closed:
* Agent A's resolver wiring to `_lens_real.py` + `_lens_traced.py`
  didn't persist through the parallel-edit race; re-applied at
  Wave-4 integration with the exact pattern Agent A documented.
* Agent G's `analysis/core.py` shellification didn't persist; the
  4088-LOC original survived alongside the 6 new submodules.
  Re-shellified at Wave-4 integration with `from .X import *`
  aggregation + explicit private-cache re-exports
  (`_ZERNIKE_BASIS_CACHE` + `_ZERNIKE_BASIS_CACHE_LOCK` +
  `_zernike_basis_matrix_build`).
* Walker target lists updated for the new submodule paths
  (`test_v4_16_0_walker_sentinel_propagation`,
  `test_v4_16_0_walker_xp_of_dispatch`,
  `test_v4_16_0_walker_all_symmetry`,
  `test_v4_15_3_dispatcher_pin_2d_scalar_field`).
* CHANGELOG line-citation refresh:
  `optimize/core.py:3032` -> `optimize/wrapper_merits.py:855`
  (`_ZERO_APERTURE_MASK` branch); `optimize/core.py:987` ->
  `optimize/merit_terms.py:515` (`MatchIdealSystem._make_source`
  `ap>0` branch); `optimize/core.py:2044-2054` ->
  `optimize/context.py:74-84` (sentinel class block).
* `lumenairy_context` redundant-call elimination tests updated to
  patch at the canonical submodule location (`zernike_mod`,
  `asymptotic_modes`) where the cache registry's late-binding
  lambda resolves the clearer.
* 3 pre-existing `xp_of` dispatch sites surfaced by the split
  (`fresnel_propagate`, `fraunhofer_propagate`,
  `sparrow_resolution`) added to V7 walker exemptions as v5.2+
  cleanup candidates.
* 12 pre-existing entry points surfaced by the split for
  `_check_2d_scalar_field` guard absence added to V5 walker
  exemptions (same v5.2+ cleanup theme).

### Test counts

Per-agent contributions (Wave-3 splits):

| Agent | Regression suite | LOC |
|---|---|---|
| A (resolver rollout) | 17 tests | 250 LOC |
| B (raytrace) | 141 tests | 325 LOC |
| C (propagation) | 122 tests | 354 LOC |
| D (asymptotic) | 85 tests | 439 LOC |
| E (optimize) | 14 tests | ~150 LOC |
| F (prescriptions) | 87 tests | 300 LOC |
| G (analysis) | 217 parametric tests | 444 LOC |

Plus Wave-4 integration fix-ups (~50 LOC across 8 test files).

### Items deferred from v5.1 to v5.2+

The v5.0 CHANGELOG's "deferred" list with one strikethrough:

* ~~Library-wide default-config knob resolver rollout~~ **shipped in
  v5.1**
* ~~6 large-file splits~~ **shipped in v5.1**
* `lumenairy.MCF` top-level alias (deferred)
* 26 formula-3 glass coefficients (deferred)
* Off-axis conic in surface frame (deferred)
* 5 new examples (deferred)
* 57-file `test_audit_fixes_*` consolidation (deferred)
* mypy CI activation (deferred -- 76 scope-local errors as of v5.1
  ship, up from 63 at v5.0.1; the v5.1.0 entry originally cited 63
  but the count had drifted -- corrected at v5.1.1, audit
  P2-NEW-F2-2)
* Ruff cosmetic-baseline cleanup -- 893 errors as of v5.1 ship, up
  from 692 at v5.0.1; same v5.1.1 correction

---

## [5.0.1] — 2026-05-20

**Closes the v5.0.0 audit (`docs/audits/AUDIT_V5_0_0_2026_05_20.md`)
through P3.**  Zero P0; 3 P1 + 5 P2 + 8 P3 across infrastructure
(lint baseline, benchmarks drift, stale "v5.0" warning text, missing
anti-regression pins, stale docstrings, ROADMAP drift).  **Zero
physics regressions in 7 consecutive releases.**  3 agents in
disjoint scopes (`A: F821 + ruff baseline`, `B: shim-removal
anti-regression pins + counter-pin`, `C: ROADMAP refresh + mypy +
P3 cluster`).

**2889 unit tests pass** (collected = 2895 = pass + 5 skip + 1
xfail), up from 2858 at v5.0.0; **+31 net** (A=4, B=6, C=21).
**34/34 validation pass.**

### P1 closures (3)

* **`set_default_*` UserWarning text updated v5.0 -> v5.1** (audit
  P1-NEW-F1-2).  At v5.0 HEAD the warning bodies in
  `propagators/propagation.py` said *"Consumer wiring at
  apply_real_lens / apply_real_lens_traced / propagate is staged
  for v5.0 alongside the file-split work."*  But the v5.0
  CHANGELOG had explicitly deferred that rollout to v5.1.  At
  v5.0 HEAD users calling `set_default_wave_propagator('fresnel')`
  saw a warning promising the bug was fixed "in v5.0" -- *which
  IS v5.0*.  The pinning test at `test_v4_16_3_agent_b.py`
  codified the misleading contract.  Fix: warning text now reads
  "staged for v5.1"; pinning test asserts `'v5.1' in msg`.
  v4.16.3's "default-knob honesty" closure is now genuinely
  honest at v5.0.1.
* **`benchmarks/test_bench_jax_jit.py` double-break fixed** (audit
  P1-NEW-V3-1).  Line 100 used `from lumenairy.system import
  propagate_through_system_jax, _PROPAGATE_SYSTEM_JAX_CACHE`
  (v5.0 `ModuleNotFoundError`); line 110 used `'params':
  {'radius': 200e-6}` (v5.0 `ValueError` on legacy aperture
  schema).  Both breaks fixed:
  `from lumenairy.propagators.system import ...` + `'params':
  {'diameter': 400e-6}`.  `benchmarks/` is not in `tests/unit/`
  CI collection scope, so the unit-CI gate didn't catch it.
* **CI lint baseline: 4 F821 real bugs fixed + advisory mode**
  (audit P1-NEW-V4-1).  `ruff check lumenairy/ tests/unit/`
  failed with 696 errors at v5.0 ship: 692 cosmetic (I001
  imports, F401 unused, F841 unused-var, F541 empty f-string,
  E702 semicolons) + **4 real F821 forward-reference bugs** in
  `lumenairy/algebra/base.py` and `lumenairy/propagators/system.py`
  (string-quoted annotations to lazily-imported `Source` /
  `PropagationResult` types missing a `TYPE_CHECKING` binding).
  Fix: F821 sites get proper `if TYPE_CHECKING:` blocks (real
  code-quality improvement, not papered over).  Lint job
  promoted to **advisory mode** (`continue-on-error: true`) at
  v5.0.1 with an inline comment noting that the cosmetic 692-
  error cleanup is a v5.1 mechanical-work item alongside the
  file splits.  PRs see lint output but don't fail-merge on it.

### P2 closures (5)

* **`simulate_detector_image` -> `apply_detector` doc-naming
  consistency** in Migration-Guide.md, CHANGELOG.md, README.md
  (audit P2-NEW-V3-2 / F1-3 2-way convergent).  The function is
  `apply_detector`; `simulate_detector_image` is not exported
  anywhere.  Users wouldn't have found the function the
  migration recipe named.
* **5 shim-removal anti-regression pins added** in
  `tests/unit/test_validation_helpers.py` (audit P2-NEW-F2-1).
  v5.0 shipped only `test_analysis_dot_analysis_shim_removed_in_
  v5_0` -- the other 4 v5.0 shim removals (`lumenairy.ao`,
  `lumenairy.io.hdf5`, top-level `lumenairy.system`, JAX aperture
  legacy schema, `cosmic_ray_rate` kwarg) had no anti-regression
  pin.  Risk: a v5.1 maintainer could accidentally re-add a
  removed shim with no test failure.  Now all 5 v5.0 honest-
  break closures have parallel pins with
  `pytest.raises(..., match=...)` that lock in the migration-
  recipe text alongside the raise.
* **ROADMAP v5.1 section refresh** (audit P2-NEW-F2-3).  v5.0's
  ROADMAP v5.1 block listed items v5.0 had already shipped (CI
  gates, public-API smoke, Python 3.10 bump, system.py move, 5
  of 8 shim removals, Migration-Guide existence).  "Read like
  the v5.0 plan, not the post-v5.0 horizon."  Stripped shipped
  items; refreshed live LOC counts for the 6 deferred file
  splits; added "Active back-compat shims at v5.0 (intentionally
  kept)" subsection documenting the 3 `apply_*_lens` re-exports
  preserved by design; refreshed the "Current state" header.
* **2 stale docstrings fixed** (audit P2-NEW-F1-4).  (a)
  `lumenairy/analysis/detector.py:82-100` still documented the
  removed `cosmic_ray_rate` kwarg as "Retained for back-compat"
  -- not retained.  Replaced with v5.0 removal note + migration
  recipe.  (b) `lumenairy/propagators/system.py:582` docstring
  example said `>>> result = la.system.evaluate(rx, src)` --
  `la.system` no longer exists.  Now `la.evaluate(...)`.
* **`[tool.mypy]` config preparation** (audit P2-NEW-V4-2).
  Added `follow_imports = "silent"` so a v5.1 mypy CI activation
  only sees the 63 scope-local errors that the cleanup actually
  owns (vs the ~1889 cascade errors from following unannotated
  downstream modules).  Activation deferred to v5.1.

### P3 closures (8)

* **CHANGELOG `__all__` arithmetic fix** (3-way V3+V4+F2
  convergent).  `len(lumenairy.__all__) == 533`; the "536" cited
  in the v5.0 CHANGELOG was the pytest case count (533
  parametrized + 3 standalone smoke tests).  Now reads "533
  entries verified via 536 smoke tests".
* **CHANGELOG `ui/` -> `lumenairy/ui/` doc drift** (P3-NEW-F2-2).
  Matches the actual `pyproject.toml` ruff `extend-exclude` value.
* **MCF `coherence_at(...)` deferral clarification** (P3-NEW-
  F2-MCF).  `PartialCoherenceMCF` + `coherence_at(...)` already
  shipped in v4.15.1 (`lumenairy/sources/core.py:1410, :1598`) and
  the class is re-exported at the top level.  What the v5.1
  deferral actually adds is the shorter `lumenairy.MCF` top-level
  alias for symmetry with `lumenairy.propagate_ensemble`.
  CHANGELOG bullet rewritten to be explicit; ROADMAP gains a
  dedicated "Partial-coherence / MCF public-API polish"
  subsection.
* **`apply_*_lens` shim preservation documented** for future-
  audit clarity.  The v5.0 work decided these re-exports are
  legitimate public API surface (not deprecation shims).
  CHANGELOG "Shims preserved" block extended with explicit
  forward-audit guidance so v5.2+ audits don't re-flag them.
* **Negative counter-pin for `test_public_api.py`** (P3-NEW-
  F1-4).  Injects a phantom name into `lumenairy.__all__`,
  asserts `hasattr(la, phantom) is False`, cleans up in
  `finally`.  Proves the smoke-test assertion machinery isn't
  vacuous.
* **V11 walker stale Python 3.9 comments refreshed**
  (P3-NEW-F1-1) at
  `test_v4_16_2_dispatcher_pin_doc_consistency.py:51-52, :68`.
  Library is 3.10+ at v5.0; comments updated.
* **Unreachable post-raise tuple-return block removed** in
  `lumenairy/propagators/system.py:932-935` (P3-NEW-F1-3).
  `_reject_legacy(...)` always raises, so the subsequent tuple
  return was unreachable.
* **Stale "one-shot deprecation warning" comment refreshed** at
  `lumenairy/propagators/system.py:1242-1248` (P3-NEW-F1-2).
  v5.0 changed the legacy aperture schema to a hard ValueError;
  the comment now reflects that.
* **Python 3.14 classifier dropped** (P3-NEW-V3-3).  CI matrix
  runs 3.10-3.13; 3.14 was aspirational.  Either drop or add to
  CI -- v5.0.1 drops with a comment that v5.1 can re-add 3.14
  alongside a CI matrix update.

### Files touched

* `lumenairy/algebra/base.py` -- TYPE_CHECKING guard
* `lumenairy/propagators/propagation.py` -- warning text v5.0 -> v5.1
* `lumenairy/propagators/system.py` -- TYPE_CHECKING guard +
  stale-comment cleanups + unreachable-code removal + docstring
  example `la.system.evaluate` -> `la.evaluate`
* `lumenairy/analysis/detector.py` -- `cosmic_ray_rate` docstring
  rewritten with v5.0 removal note
* `.github/workflows/unit-tests.yml` -- lint job advisory mode
* `pyproject.toml` -- mypy `follow_imports = "silent"`; Python
  3.14 classifier dropped; version 5.0.1
* `benchmarks/test_bench_jax_jit.py` -- v5.0 double-break fixed
* `Migration-Guide.md` -- `apply_detector` rename; §4.16.2 v5.0 ->
  v5.1 deferral
* `README.md` -- `apply_detector` rename
* `ROADMAP.md` -- v5.1 section refresh + "Current state" header
* `CHANGELOG.md` -- this entry; doc-naming fixes; arithmetic;
  MCF + `apply_*_lens` clarifications
* `tests/unit/test_v4_16_3_agent_b.py` -- pinning test v5.0 ->
  v5.1
* `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` --
  stale Python 3.9 comments refreshed
* `tests/unit/test_validation_helpers.py` -- 5 shim-removal
  anti-regression pins
* `tests/unit/test_public_api.py` -- negative counter-pin
* `tests/unit/test_v5_0_1_agent_a.py` (NEW) -- F821 / TYPE_CHECKING
  regression tests
* `tests/unit/test_v5_0_1_agent_c.py` (NEW) -- ROADMAP / mypy /
  docstring regression tests

---

## [5.0.0] — 2026-05-20

**Major release.**  v5.0 is the coordinated breaking-change release:
removes back-compat shims that had been carried 3-9 releases past
their deprecation cycle, bumps the Python floor to 3.10, moves
`lumenairy/system.py` under `propagators/` where it functionally
belongs, and adds the CI infrastructure (ruff lint, mypy strict
incremental, fast-PR unit-test gate, public-API smoke test) that
the structural cleanup needs.

**Scope discipline:** the v4.16.x ROADMAP scoped a wider v5.0 ("6
file splits + library-wide resolver rollout + MCF coherence object
+ formula-3 coefficient ingestion + off-axis conic + 5 new
examples + 57-file test consolidation").  Those non-breaking items
move to **v5.1+** so the v5.0 diff stays reviewable.  See
`ROADMAP.md` for the v5.1 horizon.

### Breaking changes

* **Python 3.10+ required.**  `requires-python = ">=3.10"` in
  `pyproject.toml`.  Python 3.9 reached EOL on 2025-10.
* **`lumenairy.system` -> `lumenairy.propagators.system`.**  The
  sequential-propagation entry points functionally ARE a
  propagator -- they walk elements applying per-element
  propagators -- not a top-level peer of `propagators/` and
  `elements/`.  Public namespace (`import lumenairy as la;
  la.propagate_through_system(...)`) unchanged.  Direct imports
  of the private path break: `from lumenairy.system import X` ->
  `from lumenairy import X` (preferred) or `from
  lumenairy.propagators.system import X`.
* **5 back-compat shims removed**:
  * `lumenairy.analysis.analysis` (v4.7 rename shim) -- now
    raises `ModuleNotFoundError`.  Use `lumenairy.analysis`.
  * `lumenairy.ao` (v4.3 shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.analysis.ao` or the
    top-level `lumenairy.DeformableMirror`.
  * `lumenairy.io.hdf5` (rename shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.io.storage` or the
    top-level re-exports.
  * `propagate_through_system_jax` legacy aperture schema
    (pre-v4.12; deprecated v4.12, removed v5.0).  Legacy params
    `radius` / `half_width_x` / `inner_radius` now raise
    `ValueError` with the canonical-schema migration recipe
    inline.  Migrate: double the value and rename
    (`radius=r` -> `diameter=2*r`, etc.).
  * `apply_detector(..., cosmic_ray_rate=...)` (v4.9
    deprecated kwarg; did not scale with detector area or
    exposure) -- removed; now raises `TypeError` (unexpected
    keyword argument).  Migrate to
    `cosmic_ray_rate_per_m2_per_s=R/A/T` where `A` is the
    detector area and `T` is the exposure time.
* **Shims preserved as legitimate public API surface** (not
  removed despite the ROADMAP's audit-V4_13_1 suggestion):
  * `lumenairy.elements.lenses.apply_*_lens` re-exports.  These
    provide a coherent one-stop import surface; the underlying
    file-split into `_lens_thin.py` / `_lens_real.py` /
    `_lens_traced.py` is an internal organisational choice
    rather than a deprecation cycle.
    **Note for future audits (v5.0.1 audit
    `apply_*_lens` shim-preservation closure)**: these
    re-exports are **intentionally retained** as the canonical
    one-stop user-facing import path -- a v5.2+ audit that
    flags them as "stale shim removable" should be rejected
    with a pointer to this CHANGELOG entry.  The decision was
    made at v5.0 ship after weighing the v4.13.1 ROADMAP
    suggestion against the user-facing ergonomics of the
    single ``from lumenairy.elements.lenses import apply_real_lens``
    import; the latter won.

### CI gates + infrastructure

* **NEW `.github/workflows/unit-tests.yml`** -- fast PR feedback
  gate.  Runs `pytest tests/unit -m "not integration"` on Python
  3.10, 3.11, 3.12, 3.13; `ruff check` on the library + unit
  tests; the new public-API smoke test.
* **NEW `[tool.ruff]` config** in `pyproject.toml`.  Conservative
  initial rule set (E, F, I) with documented per-file ignores;
  excludes `validation/`, `docs/`, `examples/`, `lumenairy/ui/`
  from the v5.0 baseline (v5.0.1 audit P3-NEW-F2-2 closure of
  the `"ui/"` -> `"lumenairy/ui/"` doc drift).
* **NEW `[tool.mypy]` config** -- incremental adoption starting
  with the small self-contained modules (`backend/`,
  `_deprecation.py`, `_context.py`, `progress.py`, `memory.py`).
  Everything else stays untyped for v5.0.
* **NEW `tests/unit/test_public_api.py`** -- asserts every name
  listed in `lumenairy.__all__` is resolvable via
  `getattr(lumenairy, name)`.  Catches "exported but not
  imported" / "imported but not exported" sibling-gap at the
  facade.  533 entries verified via 536 smoke tests
  (533 parametrized + 3 standalone) at v5.0 ship.

### Migration-Guide.md

`Migration-Guide.md` adds a v5.0.0 section with concrete
old->new recipes for each breaking change.  The deferred v5.1+
items are listed honestly so users know what to expect.

### Tests + CI

* **2858 unit pass / 5 skip / 1 xfail = 2864 collected** (was
  2327 at v4.16.3; +531 net -- the v5.0 work landed alongside
  cumulative v4.16.x test additions in this session).
* **34/34 validation pass.**
* Updated callers across `lumenairy/` and `tests/` to the new
  `lumenairy.propagators.system` import path.
* Updated v4.12 aperture-schema tests + v4.9 cosmic_ray_rate
  test from "must warn" to "must raise" semantics.

### Deferred from v5.0 to v5.1+ (see ROADMAP.md)

* 6 large-file splits (`raytrace/core.py`,
  `propagators/propagation.py`, `propagators/asymptotic.py`,
  `optimize/core.py`, `io/prescriptions.py`,
  `analysis/core.py`).  Pure mechanical reorganisation; no
  public API change.
* Library-wide default-config knob resolver rollout (`set_default_
  wave_propagator`, `set_default_dy`, `set_default_real_dtype`
  remain API-only at v5.0; the v4.16.3 one-shot UserWarning stays
  in place).
* MCF top-level public-API polish (v5.0.1 audit P3-NEW-F2-MCF
  clarification).  `PartialCoherenceMCF` -- including its
  `coherence_at(...)` two-point query -- already shipped in v4.15.1
  (`lumenairy/sources/core.py`) and is re-exported at the
  top level as `lumenairy.PartialCoherenceMCF` since v4.15.1.  What
  v5.1 still owes is the *naming* polish: a shorter top-level alias
  `lumenairy.MCF` for symmetry with `lumenairy.propagate_ensemble`
  / `lumenairy.coherence_at` so the "import the canonical name"
  story is uniform across the partial-coherence surface.  The
  v4.16.x ROADMAP entry "MCF object" predates the v4.15.1 ship and
  was carried forward without rewording at v5.0; this CHANGELOG
  bullet is the authoritative deferral statement.
* Off-axis conic in surface frame.
* 26 formula-3 (polynomial) glass coefficients.
* 5 new examples (multi-config / zoom, tolerancing, coronagraph
  workflow, AO closed-loop, ghost / stray-light).
* 57-file `test_audit_fixes_*` consolidation into topical homes.

---

## [4.16.3] — 2026-05-20

**Closes the v4.16.2 audit
(`docs/audits/AUDIT_V4_16_2_2026_05_20.md`) through P3.**  Audit
found zero P0 + 2 P1 + 6 P2 + 8 P3, concentrated around v4.16.2's
pre-v5.0 prep features being mostly scaffolding without real
consumers, plus 2 structural-bypass issues inside the new V11
doc-consistency walker (the very walker designed to retire the
documentation-surface sibling-gap meta-pattern contained that
pattern itself).  4 agents in disjoint scopes (`A: V11 walker
hardening`, `B: default-knob consumer wiring + Migration-Guide
correction`, `C: optimize/core.py P2 polish`, `D: P3 cluster +
soften "structurally retired" claim`).

**2327 unit tests pass** (collected = 2333 = pass + 5 skip + 1
xfail), up from 2270 at v4.16.2; **+57 net** (per-agent: A=17,
B=16, C=8, D=15, walker-extra=1 = 57).  **34/34 validation pass**.

### P1 closures (2)

* **V11 walker pyproject parsing -> `tomllib`** (Agent A; audit
  P1-NEW-F1-2).  The v4.16.2 11th meta-pin walker used the regex
  `r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\[(.*?)\]'` -- non-greedy,
  stopping at the first `]`.  Silently mis-parsed
  `jax-gpu = ["jax[cuda12]>=0.4.20"]` (captured only `"jax[cuda12`)
  and the `all = [...]` block when its body contained
  ` `lumenairy[all]` ` comment-bracket text.  The walker passed
  vacuously because `refractiveindex` was independently captured
  via `[glass]`; if a future maintainer removed the dedicated
  `[glass]` group keeping the dep only in `[all]`, drift would
  go green WITH drift.  Replaced regex parsing with `tomllib`
  (Python 3.11+) / `tomli` (3.9/3.10 backport) graceful fallback.
  Two anti-regression pins assert the literal v4.16.2 strings no
  longer appear in the walker source.
* **Migration-Guide.md §4.16.2 corrected** (Agent B; audit
  P1-NEW-F1-1).  The v4.16.2 recipe used
  `set_default_wave_propagator('fresnel')` followed by
  `apply_real_lens(...)` -- but `apply_real_lens` hardcodes
  `wave_propagator: str = 'asm'` and does NOT consult the default
  knob.  A user copy-pasting silently used ASM.  Rewrote the §4.16.2
  section: explicit "API-only in v4.16.2/v4.16.3" limitation note;
  replaced the misleading recipe with one using `set_default_complex_
  dtype` + `set_default_real_dtype` (knobs with real consumers);
  retained `wave_propagator=` per-call kwarg on the apply_real_lens
  example.

### P2 closures (6)

* **`get_default_real_dtype` consumer wiring fixed** (Agent B;
  audit P2-NEW-F1-3).  v4.16.2's "representative wiring" at
  `propagators/ensemble.py:347-355` was structurally unreachable
  dead code -- the `except (TypeError, ValueError)` branch could
  never fire because the earlier shape check at `:308-330` already
  guaranteed `ensemble.dtype` was a valid numpy dtype.  Refactored
  so `get_default_real_dtype()` is the canonical `in_dtype is None`
  fallback path (now reachable via the `getattr(ensemble, 'dtype',
  None)` default).  The knob is now narrowly consumed at one
  reachable site, preserving the v4.16.2 CHANGELOG claim of at least
  one wired consumer.
* **`set_default_wave_propagator` + `set_default_dy` no-consumer
  `UserWarning`** (Agent B; audit P2-NEW-F1-4).  Both knobs store
  values no library code reads at v4.16.3.  Setters now emit a
  one-shot module-level-latched `UserWarning` informing users that
  the knob is "API-only at v4.16.2/v4.16.3; consumer wiring at
  `apply_real_lens` / `apply_real_lens_traced` / `propagate` lands
  in v5.0".  Sibling-gap pin asserts these knobs have zero
  consumers library-wide -- when v5.0 adds the first consumer, the
  pin FAILS LOUDLY prompting removal of the stale warning + the
  Migration-Guide limitation note.
* **V11 version list -> CHANGELOG-driven** (Agent A; audit
  P2-NEW-F2-MED-1).  v4.16.2's `test_migration_guide_has_known_
  version_sections` hardcoded `('4.13.0', '4.15.1', '4.16.1',
  '4.16.2')` -- when v4.17.0 ships with a breaking change, walker
  would pass silently unless someone manually edited the tuple.
  Replaced with `_versions_with_breaking_changes_from_changelog()`
  scan extracting `## [X.Y.Z]` headings; high-precision markers
  only (`silent semantics change`, `SUM->AVG`, etc.) plus
  `### Breaking changes` heading detection; documented `_MIGRATION_
  GUIDE_SIBLING_COVERED` allowlist for v4.15.2 (its migration recipe
  lives under v4.15.1's Migration-Guide section).
* **V11 extends to CHANGELOG↔Migration-Guide drift coverage**
  (Agent A; audit P2-NEW-F2-MED-2).  New
  `test_migration_guide_sections_are_non_trivial` test enforces
  each `## X.Y.Z` section has >=200 chars of non-whitespace body.
  Future CHANGELOG entries flagged "breaking" must come with a
  substantive Migration-Guide entry or the walker fails.
* **`Constraint` auto-probe DeprecationWarning** (Agent C; audit
  P2-NEW-F1-1).  v4.16.1 shipped a `Constraint.__post_init__`
  auto-probe; v4.16.2 silently removed it.  v4.16.3 emits a
  one-cycle DeprecationWarning via module-level latch, pattern-
  parallel to the v4.16.2 `MultiWavelengthMerit` `FutureWarning`
  latch.  Scheduled for removal in v5.0.
* **`pickle.dumps` probe catch widened** (Agent C; audit
  P2-NEW-F1-2).  v4.16.2's `except (pickle.PicklingError,
  AttributeError, TypeError)` missed `RecursionError` (deep object
  graph), `RuntimeError` (custom `__reduce__`), `MemoryError`,
  and arbitrary `__reduce__` / `__getstate__` exceptions.  Widened
  to `except Exception` -- pickling is best-effort heuristic; any
  failure is "not safely picklable" signal.  `BaseException`
  (`KeyboardInterrupt` / `SystemExit`) intentionally still
  propagates.

### P3 closures (8)

* **`__polynomial__` sentinel** parallel to `__sellmeier__` (Agent D;
  audit P3-NEW-F1-1).  `POLYNOMIAL_COEFFICIENTS` dispatch was
  fallback-only when refractiveindex was unavailable; with the
  sentinel, polynomial-formula glasses can opt in to the bundled
  evaluator even with refractiveindex installed.  Extended
  `_check_glass_registry_consistency` with forward + reverse
  polynomial checks; `get_glass_index_complex` updated to include
  `__polynomial__` in the no-extinction sentinel tuple.
* **POLYNOMIAL/SELLMEIER dispatch order doc/code reconcile**
  (Agent D; audit P3-NEW-V3-1).  Code does SELLMEIER -> POLYNOMIAL;
  v4.16.2 docs claimed the opposite.  Inline comment added citing
  the actual order.
* **`DEFAULT_*` constants re-exported at top level** (Agent D;
  audit P3-NEW-F2-LOW-1).  `DEFAULT_COMPLEX_DTYPE` was already
  exported via `lumenairy/__init__.py`; the v4.16.2 new globals
  (`DEFAULT_REAL_DTYPE`, `DEFAULT_WAVE_PROPAGATOR`, `DEFAULT_DY`)
  were not -- sibling-gap.  Added to both the import block and
  `__all__`.
* **Per-surface (not max) thickness in high-NA hoist message**
  (Agent D; audit P3-NEW-F1-4).  `_maybe_warn_transfer_jax_high_na`
  now accepts `surface_index`; hoist loops surfaces to find the
  worst |N| and cites THAT surface's thickness in the user-facing
  message (was overstating worst-case drift via `max(thickness)`).
* **Multiprocess / fork-safety documentation** (Agent D; audit
  P3-NEW-F1-2 + P3-NEW-F1-3).  Added a "Multiprocess / fork notes"
  section near the top of `propagators/propagation.py` documenting
  that the one-shot latches AND the `DEFAULT_*` module-level globals
  are NOT pickle/fork-safe -- spawn-mode workers re-import the
  module and reset to defaults / re-emit warnings.  Not fixing the
  semantics (would need shared-state); just documenting honestly.
* **`psutil` promoted to Required in requirements.txt** (Agent D;
  audit P3-NEW-F1-5).  `psutil>=5.0` is a hard dep in
  pyproject.toml but the v4.16.2 requirements.txt listed it under
  "Recommended".  Promoted for parity.
* **"Structurally retired" claim softened** (Agent D; audit
  P3-NEW-F2-LOW-2).  CHANGELOG + ROADMAP both said "structurally
  retired across all known classes"; honest framing is "retired
  across all currently-known classes; new classes will continue to
  surface".
* **CHANGELOG sentinel line citation refresh** `:3015` -> `:3032`
  (~17 lines added by Agent C's `Constraint` DeprecationWarning
  latch + pickle catch widening).

### Tests + CI

* **2327 pass / 5 skip / 1 xfail = 2333 collected** (up from
  2270 / 5 / 1 = 2276 at v4.16.2; +57 net).  Per-agent breakdown:
  A=17, B=16, C=8, D=15, walker-extra=1.  Sum: 57.
* New test modules:
  * `tests/unit/test_v4_16_3_agent_a.py` (17 tests)
  * `tests/unit/test_v4_16_3_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_3_agent_c.py` (8 tests)
  * `tests/unit/test_v4_16_3_agent_d.py` (15 tests)
* V11 walker grew from 7 to 8 tests (`test_migration_guide_sections_
  are_non_trivial` added).

---

## [4.16.2] — 2026-05-20

**Closes the v4.16.1 audit
(`docs/audits/AUDIT_V4_16_1_2026_05_19.md`) through P3** -- a focused
follow-up after v4.16.1 hit PyPI.  Audit found zero P0 / 5 P1 / 8 P2
/ 9 P3, concentrated in (a) 3 code-correctness items the v4.16.1
verifier audit missed because the test pins themselves bypassed the
production path, and (b) 4 documentation-surface drifts that proved
the sibling-gap meta-pattern had migrated from code surfaces
(covered by V1-V10 walkers) to documentation surfaces (uncovered).
4 agents in disjoint scopes (`A: JAX gate + ensemble dispatch`,
`B: optimize/core.py`, `C: pre-v5.0 features + glass P3`,
`D: doc-surface + 11th walker + Migration-Guide`).

**Also lands the user-requested pre-v5.0 prep features**:
* Bundled Sellmeier formula-3 (polynomial) evaluator infrastructure
* 3 library-wide default-config knobs (`set_default_real_dtype`,
  `set_default_wave_propagator`, `set_default_dy`)
* `Migration-Guide.md` skeleton at the repo root

**2270 unit tests pass** (collected = 2276 = pass + 5 skip + 1
xfail), up from 2198 at v4.16.1; +78 net (per-agent: A=18, B=16,
C=24, D=20 = 78 -- reconciles exactly).  **34/34 validation pass**.

### P1 closures -- code findings (Agent A + Agent B)

* **`_transfer_jax` high-NA warning structurally unreachable
  in production** (Agent A; audit P1-NEW-F1-1).  The v4.16.1 gate
  at `jax_trace.py:579` used `isinstance(direction_n, np.ndarray)`,
  but `make_jax_ray_state(...)` calls `jnp.asarray(N)` which yields
  `jax.Array` -- NOT a `np.ndarray` subclass since JAX 0.4+.  The
  gate returned early on every production user-flow call.  Fix:
  duck-typed gate (`np.asarray(direction_n)` probe; rejects
  `jax.core.Tracer`), PLUS an eager-only one-shot probe hoisted
  to the entry of `trace_jax` BEFORE the inner `jax.jit` wrapper
  (the jit wrapper makes everything inside a Tracer, regardless of
  the gate).  New integration test calls `trace_jax(...)` with
  `make_jax_ray_state(N=0.5*np.ones(K))` end-to-end and asserts the
  RuntimeWarning fires -- closes the production-path gap.
* **`propagate_ensemble` silently downcasts CuPy/JAX to NumPy**
  (Agent A; audit P1-NEW-F1-2).  `ensemble.py:253`'s
  `np.asarray(ensemble)` triggered GPU->CPU transfer (CuPy) or
  forced concretization (JAX), defeating the docstring's "tolerate
  duck-typed array protocols" claim.  Fix: `array_namespace`
  dispatch via `lumenairy.backend`; accumulator built on the
  matching `xp` so the GPU / JAX paths stay on the backend.  Also
  rewrites `_coerce_field_from_propagator_return` to preserve
  backend.
* **`MultiWavelengthMerit` SUM->AVG one-cycle `FutureWarning`**
  (Agent B; audit P1-NEW-F1-3).  v4.16.1's SUM->AVG fix was correct
  but silent -- existing user-calibrated 3-wavelength configs
  silently dropped 3x.  v4.16.2 emits a one-cycle `FutureWarning`
  via module-level latch when `len(wavelengths) > 1`, alerting
  users to re-scale `weight` by `len(wavelengths)` if they tuned
  against pre-v4.16.1 SUM behavior.  Latch ensures the warning
  fires only ONCE per process (critical -- without the latch the
  warning would flood optimization loops).
* **README -> pyproject.toml dependency declaration sync**
  (Agent D; audit P1-NEW-F2-HIGH-1).  README's `### Required`
  block still listed `refractiveindex` as Required + `pip install
  numpy refractiveindex` as the quick-install command; pyproject
  moved it to `[glass]` extras in v4.16.1.  Full dependency block
  rewritten: enumerates each pyproject extras group + `pip install
  lumenairy[glass]` as the canonical install pattern.
* **requirements.txt -> pyproject.toml sync** (Agent D; audit
  P1-NEW-F2-HIGH-2).  Dropped uncommented `refractiveindex>=1.0`;
  moved `h5py>=3.0` to commented section (it's only in `[hdf5]` /
  `[gui]` extras); updated commented `zarr>=2.14` -> `zarr>=3.0`
  to match v4.16.1's floor bump.  Added commented lines for every
  optional-extras group + header note pointing at pyproject.toml
  as the canonical source.

### P2 closures (8) -- API consistency + meta-pins + doc hygiene

* **`Constraint.__post_init__` probe -> opt-in `.validate()` method**
  (Agent B; audit P2-NEW-F1-1).  v4.16.1's probe ran real user code
  on instantiation (e.g. BFL constraint calling `system_abcd()` on
  every `Constraint(...)` call).  Moved to opt-in `Constraint.
  validate()` method; users who relied on the auto-probe call it
  explicitly.
* **Lambda warning -> `pickle.dumps` probe** (Agent B; audit
  P2-NEW-F1-2).  v4.16.1's `__name__ == '<lambda>'` check missed
  closures (`def inner(x): ...`) and `functools.partial(lambda,
  ...)`.  Replaced with `pickle.dumps(self.fun)` probe; catches all
  unpicklable callables.
* **Existing v4.16.0 Constraint tests updated to module-level
  functions** (Agent B; audit P2-NEW-F1-3).  Five lambda-Constraint
  test sites in `test_v4_16_0_agent_c.py` migrated to module-level
  `_sum_constraint` / `_first_coord` so the v4.16.1 lambda warning
  doesn't pollute the warning channel.
* **10th meta-pin walker hardened** (Agent D; audit P2-NEW-F1-4).
  `_module_has_register_cache_clearer_call` rewritten to require
  the call appear at **module level**, not nested inside a
  function / class body / always-False `if` branch.  Canonical
  top-level `try/except ImportError` enrollment idiom still
  accepted.  4 new counter-pins (positive + negative) verify the
  tightening.
* **`_clear_local_asm_caches` late-binding-lambda registration**:
  already landed in v4.16.1 (Agent C scope at that release).
* **ROADMAP V9 -> V11 meta-pin enumeration** (Agent D; audit
  P2-NEW-F2-MED-1).  "ALL 9 dispatcher meta-pins" -> "ALL 11
  dispatcher meta-pins"; V10 + V11 entries added.  Updated the
  sibling-gap retirement claim to cover documentation surfaces too.
* **CHANGELOG test-count arithmetic** (Agent D; audit
  P2-NEW-F2-MED-2).  v4.16.1 headline `2208 / +102 / 2106`
  refreshed to `2198 / +85 / 2113` (collected metric, arithmetic
  reconciles: 2113 + 85 = 2198 = pass=2192 + skip=5 + xfail=1).
  Also corrected the Tests + CI tail section.  v4.16.2 audit note
  added explaining the discrepancy.
* **CHANGELOG `UserWarning` -> `RuntimeWarning` typo** (Agent D;
  audit P2-NEW-V2-1).  The v4.16.1 entry's High-NA `_transfer_jax`
  block said "emits a `UserWarning`" but implementation + tests
  both use `RuntimeWarning`.

### P3 closures (9)

* `propagate_ensemble` empty 3-D ensemble -> `ValueError` (Agent A)
* `propagate_ensemble` `dx`/`wavelength` kwargs collision -> clear
  `ValueError` (Agent A)
* `_resolve_bound` 3-tuple guard (Agent B)
* `Constraint.fun` docstring example: lambda -> module-level
  function (Agent B)
* LM bounds `lm` -> `trf` override UserWarning added (Agent B)
* `jax.grad` pin for B.4 dtype probe (Agent A)
* `propagator_kwargs` precedence pin for B.1 (Agent A)
* `GLASS_VALIDITY` consistency check accepts numpy scalars
  (Agent C; audit P3-NEW-F1-4)
* CHANGELOG sentinel line citation refresh `:2974` -> `:3015`
  (Agent D)

### NEW -- 11th meta-pin walker (doc-consistency)

`tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` (7
tests): closes the v4.16.1-identified documentation-surface
sibling-gap meta-pattern.  Scans 4 surfaces for drift vs the
canonical `pyproject.toml`:
* README.md `Required` block doesn't list optional-extras packages
* README.md `pip install` command doesn't force optional-extras
* requirements.txt uncommented lines match pyproject hard deps
* ROADMAP.md `ALL N meta-pins` count matches V-enumeration
* CHANGELOG.md v4.16.1 headline arithmetic reconciles
* Migration-Guide.md exists with known version sections

The sibling-gap meta-pattern is now structurally retired at BOTH
code surfaces (V1-V10) AND documentation surfaces (V11) across all
currently-known classes; new classes will continue to surface and
be added to the V-walker family as identified.

### NEW -- Pre-v5.0 prep features (Agent C)

* **Bundled Sellmeier formula-3 (polynomial) evaluator**.
  `lumenairy/glass.py`:
  - NEW `_polynomial_index(wavelength_m, coeffs, glass_name=None)`
    -- implements refractiveindex.info formula-3:
    `n^2 = c0 + sum_i c_i * lam_um ** exp_i`.  Subsumes the Schott
    6-coefficient polynomial form.
  - NEW `POLYNOMIAL_COEFFICIENTS = {}` -- empty at ship.  v4.16.2
    lands the evaluator + dispatch wiring; populating the 26
    catalogue entries (Hikari E-/J-, Sumita K-, 4 CDGM polynomial)
    requires per-glass vendor-source review + 5e-5 n_d cross-check
    against refractiveindex.info YAML and is staged for v5.0.
  - `get_glass_index` dispatch updated: when refractiveindex is
    unavailable AND the glass is in POLYNOMIAL_COEFFICIENTS,
    dispatches to `_polynomial_index` before raising ImportError.
* **3 default-config knobs**, parallel to existing
  `set_default_complex_dtype`:
  - `set_default_real_dtype(dtype)` / `get_default_real_dtype()`
    -- accepts `np.float32` / `np.float64`.
  - `set_default_wave_propagator(name)` / `get_default_wave_
    propagator()` -- accepts `'asm'`, `'sas'`, `'fresnel'`,
    `'rayleigh_sommerfeld'`, `'rs'`.
  - `set_default_dy(value)` / `get_default_dy()` -- accepts
    `None` (means "match dx") or a positive finite float.
  - All 6 functions exported at top level via `lumenairy/__init__.py`.
  - Representative consumer wiring landed in
    `propagators/ensemble.py` (no-input-dtype real fallback path
    honours `get_default_real_dtype()`).  Full library-wide
    resolver rollout staged for v5.0.
* **`Migration-Guide.md` skeleton** at repo root.  Version-spanning
  migration guide for v4.x; sections for v4.13.0 (rcwa.py rename,
  wavelength-required), v4.15.1 (Schell ensemble return shape),
  v4.16.1 (MultiWavelengthMerit SUM->AVG, refractiveindex
  optional), v4.16.2 (default-config knobs).  Forward section for
  v5.0 itemizing planned migration points.

### Tests + CI

* **2270 pass / 5 skip / 1 xfail = 2276 collected** (up from 2198
  at v4.16.1; +78 net).  Per-agent breakdown: A=18, B=16, C=24,
  D=13+7=20.  Sum: 18+16+24+20 = 78 (reconciles).
* New test modules:
  * `tests/unit/test_v4_16_2_agent_a.py` (18 tests)
  * `tests/unit/test_v4_16_2_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_2_agent_c.py` (24 tests)
  * `tests/unit/test_v4_16_2_agent_d.py` (13 tests)
  * `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py`
    (7 walker tests -- the 11th meta-pin)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch
  site drifted `:2974` -> `:3015` after Agent B's
  `MultiWavelengthMerit` `FutureWarning` latch + `Constraint`
  probe move + lambda pickle-probe (~41 lines added above the
  sentinel branch).

---

## [4.16.1] — 2026-05-19

**Closes the v4.16.0 deep audit
(`docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md`) through P3.**  The
audit was the first "deep" audit to actively hunt silent-wrong-answer
correctness bugs alongside the usual structural/UX cleanup; 4 real
physical-correctness defects surfaced, plus the previously half-shipped
Schell-model partial-coherence cluster, plus 8 hygiene items.  4 agents
worked in disjoint scopes (`A: correctness bugs`, `B: Schell + JAX
paths`, `C: constraints + meta-pins + warn hygiene`, `D: glass + compat
+ UX`).  **2198 unit tests pass** (up from 2113; +85 net) -- of
those 2198, 2192 actively pass + 5 documented skips (4 pymoo +
1 ZARR_MKDIR_PATCH) + 1 documented xfail.
**34/34 validation files pass**.  (v4.16.2 audit P2-NEW-F2-MED-2
correction: pre-v4.16.2 headline cited 2208 / +102 / baseline 2106 --
off by 10 / +17 / -7; the corrected numbers (collected = pass +
skip + xfail) reconcile to the empirical per-agent breakdown
A=11 + B=26 + C=20+6 + D=22 = 85.)

### P0 / P1 closures — correctness bugs (Agent A)

Four real silent-wrong-answer defects at user-relevant configurations
that the prior verification-style audits missed.  Each ships with an
empirical regression test pinning the failure mode and a sibling-gap
sweep confirming no other sites carry the same pattern.

* **Bug 1 — `MultiWavelengthMerit.evaluate` SUM -> AVG.**
  `lumenairy/optimize/core.py`: the wrapper's tail `return self.weight
  * total` summed sub-merit results across wavelengths rather than
  averaging.  Documented semantics + both sibling classes
  (`MultiFieldMerit`, `ToleranceAwareMerit`) divide by `len(...)` at
  the return; the bug was localised to this one class.  Fixed:
  `return self.weight * total / max(len(self.wavelengths), 1)`.  A
  3-wavelength merit now returns the same value as a 1-wavelength
  merit on the same sub-merit + constant field (was returning `3x`).
* **Bug 2 — `shack_hartmann` wavefront pitch quantisation.**
  `lumenairy/analysis/detector.py`: the slope-to-wavefront integration
  multiplied the cumsum by the user-requested `lenslet_pitch`, but the
  on-grid pitch is the integer-pixel quantised `sa_pixels * dx`.  At
  `lenslet_pitch / dx = 1.7` (`sa_pixels = 2`), the reconstructed
  wavefront amplitude was biased by `8.5 / 10 = 0.85` (17.6% low
  relative to the post-fix amplitude).  Fix: use `pitch_actual =
  sa_pixels * dx` for the cumsum step.  Empirically pinned by
  `test_bug2_shack_hartmann_amplitude_ratio_physics_pin`.
* **Bug 3 — `_detect_backend` directory misclassification.**
  `lumenairy/io/storage.py`: the auto-detect routed *any* directory
  path to Zarr regardless of whether a Zarr store was actually present
  (`if path.endswith('.zarr') or os.path.isdir(path)`).  A bare
  directory matching a typical HDF5 sibling layout was silently
  misrouted, and `pathlib.Path` callers hit `AttributeError` on the
  string-only `.endswith` check.  Fix: `str(path)` cast +
  directory-routing restricted to actual Zarr stores via the canonical
  `zarr.json` (v3) / `.zarray` (v2) marker files.
* **Bug 4 — LM `bounds` parser accepts `None` endpoints.**
  `lumenairy/optimize/core.py`: the `method='lm'` branch built
  `lb`/`ub` arrays via `b[0] if b else -np.inf`; `b = (None, 1.0)` is
  truthy, so `None` leaked into `np.array([None, 0.0, ...])` (object
  dtype), and scipy raised an opaque downstream error.  Fix: explicit
  `_resolve_bound` helper that routes any `None` (per-side or
  per-tuple) to `+/-np.inf` and produces a clean `float64` array.

### P1 closures — half-shipped clusters (Agent B)

* **`propagate_ensemble(...)` helper added** (audit Part 5 P0-1).
  New module `lumenairy/propagators/ensemble.py`, exported at the
  top-level as `lumenairy.propagate_ensemble`.  Iterates a Schell-family
  `(n_realizations, Ny, Nx)` ensemble through any coherent propagator
  (`'asm'` / `'fresnel'` / `'fraunhofer'` / `'rs'` / `'sas'` or a
  user-supplied callable) and returns `I_partial = <|E_k|^2>_k` (the
  canonical Wolf-coherence-theory result).  `return_ensemble=False` by
  default for memory efficiency; opt-in `return_ensemble=True` for the
  full propagated stack.  Shape-mismatch + 2-D-field-instead-of-
  ensemble cases raise informative `ValueError`s.  New example
  `examples/06_schell_propagation.py` measures a `~6.95x` smoothing
  factor (coherent-peak / partial-peak) on a 256x256 grid at
  `sigma_g / w0 = 0.3`, consistent with the Wolf-Carter far-field
  scaling.
* **Default Schell-factory `DeprecationWarning` retired.**  The 3
  top-level factories (`create_gaussian_schell_source`,
  `create_schell_model_source`, `create_annular_incoherent_source`)
  and 2 `Source.*` classmethods now default `return_kind='ensemble'`
  directly.  The `_RETURN_KIND_UNSET` sentinel + warning helper are
  preserved as deprecated public symbols (the v4.15.3 sentinel-
  promotion meta-pin imports them); targeted for removal in v5.0.
* **MCF rejection message refreshed.**  `lumenairy/_validation.py`:
  the "MCF planned for v4.16+" wording was stale at v4.16.0.  Updated
  to cite v5.0+ and point callers at the new `propagate_ensemble`
  helper for the partial-coherence path that lands now.
* **JAX-traceable dtype probe.**  `lumenairy/system.py` ~line 1184:
  swapped the `np.asarray(E_in).dtype` probe for duck-typing
  `getattr(E_in, 'dtype', None)` so the `propagate_through_system_jax`
  path no longer breaks under `jax.jit` / `jax.grad` tracers (which
  refuse the `np.asarray` cast).
* **High-NA `_transfer_jax` RuntimeWarning.**
  `lumenairy/raytrace/jax_trace.py`: added an eager-only guard
  (`isinstance(direction_n, np.ndarray)`-gated) that emits a
  `RuntimeWarning` when `min |N| < 0.95` — the regime where the
  paraxial small-angle approximation preserved for autodiff
  stability begins to diverge from the NumPy reference.  Tracer-time
  path is unchanged (no warning, preserves `jit` / `grad` purity).
  (v4.16.2 audit P2-NEW-V2-1: pre-v4.16.2 bullet said "`UserWarning`"
  -- code + tests use `RuntimeWarning`; corrected.)  (v4.16.2 audit
  P1-NEW-F1-1: pre-v4.16.2 the `isinstance(np.ndarray)` gate was
  structurally unreachable in production because `make_jax_ray_state`
  converts to `jax.Array` which is not a `np.ndarray` subclass since
  JAX 0.4+; v4.16.2 replaces the gate with a duck-typed
  `np.asarray(...)` probe + hoists an eager-only check to the
  `trace_jax` entry before the inner `jax.jit` wrapper.)

### P2 closures — API consistency + meta-pins (Agent C)

* **`Constraint` API narrowed to scalar-return.**  Vector-return
  callables crashed inside the pymoo wrapper's `float(_f(xv))` coercion
  with an opaque `TypeError`.  Docstring narrowed to `f(x) -> scalar`
  and `__post_init__` adds a best-effort scalar-shape probe.  3-test
  pin block exercises both the accept (scalar) and reject (ndarray of
  shape `(K,)`) paths.
* **`Constraint(fun=lambda x: ...)` UserWarning for parallel workers.**
  Lambdas aren't picklable; `differential_evolution(workers>1)` /
  joblib-parallelised FD-gradient fails with `PicklingError`.  Soft
  heads-up at `__post_init__` when `fun.__name__ == '<lambda>'`;
  single-process SLSQP / trust-constr is unaffected.
* **`_clear_local_asm_caches` late-binding-lambda registration.**
  `lumenairy/propagators/propagation.py:~803`: the cache registry
  enrollment used an early-binding partial, diverging from the
  canonical late-binding-lambda pattern of the other 8 cache
  enrollments.  Switched to `lambda: _clear_local_asm_caches()` for
  pattern parity.
* **10th cache-registry meta-pin walker.**
  `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`:
  AST-walks every `@lru_cache`-decorated module-level function and
  asserts a paired `_cache_registry` enrollment.  15 caches discovered,
  0 orphans — closes the V4-bucket sibling-gap structurally
  (continuing the V1-V9 meta-pin trajectory).
* **`_check_glass_registry_consistency()` extends to GLASS_VALIDITY.**
  Every `GLASS_VALIDITY` key now must appear in `GLASS_REGISTRY` (and
  must be a `(lambda_min, lambda_max)` 2-tuple with `lambda_min <
  lambda_max`, both finite, non-negative).
* **`warnings.warn(..., stacklevel=2)` hygiene** added at the 2 sites
  flagged in `lumenairy/io/prescriptions.py` (lines ~1019 / ~1470).

### P3 closures — compat + glass / materials + UX (Agent D)

* **4 stale `n_d` inline comments in `lumenairy/glass.py` corrected**
  to match the actually-computed Sellmeier values.  Multi-way
  convergent finding across audit perspectives (V5 + DEEP-3 + F1 +
  PHYS-2):
  * H-ZK9B: `1.613750` -> `1.62041`
  * H-ZF12: `1.673000` -> `1.76182`   (the most egregious — 6% off)
  * D-LAK52: `1.729160` -> `1.73050`
  * H-ZLAF52A: `1.796800` -> `1.80610`
  No runtime behaviour change — the actual Sellmeier coefficients were
  always correct; only the comments were stale.
* **`refractiveindex` moved to optional `[glass]` extras group.**
  `pyproject.toml`: dropped from hard `[project.dependencies]`,
  promoted to `glass = ["refractiveindex>=1.0"]`, and bundled into the
  existing `all` group.  Aligns the wheel with the lazy-import +
  `SELLMEIER_COEFFICIENTS` fallback already in place in
  `lumenairy/glass.py`.
* **`zarr>=2.14` floor bumped to `zarr>=3.0`** in the `zarr` and `all`
  extras groups.  `lumenairy/io/storage.py` uses `Group.create_array`
  (a Zarr v3 API); the v2 floor was a latent `AttributeError` waiting
  for any zarr=2.x user.
* **`ProcessPoolExecutor` `spawn` mp_context.**
  `lumenairy/elements/_lens_traced.py:~220`: explicit
  `mp_context=multiprocessing.get_context('spawn')` kwarg on the
  module-level worker pool.  Matches the README + v4.16.0 CHANGELOG
  claim that `spawn` is used (previously `fork` on Linux — unsafe
  with cached FFT plans and worker threads).
* **`examples/06_schell_propagation.py`** + **`examples/07_zemax_load_trace.py`** added.  The Zemax-loader example
  closes audit UX item 22 (no prior example exercised
  `la.load_zemax_zmx` end-to-end); loads the Thorlabs AC254-100-C
  achromat fixture and falls back to a programmatic N-BK7 singlet via
  `la.make_singlet` if the .zmx file is missing.
* **`CONVENTIONS.md`** added at the repo root — ~10 short sections
  documenting the `create_*` (-> field/Source) vs `make_*` (->
  prescription / bundle / non-field) factory verb contract, error-
  message prefix discipline, RNG kwarg name, and 7 related
  conventions that previously lived only in informal precedent.
* **Stray repo-root artifacts cleaned up.**  3 example PNGs moved
  from the repo root to `examples/output/` (the 3 producing scripts
  updated to write there); stray `C:tmpoptimize_diff.txt` echo-
  redirect artifact deleted (same cleanup pattern as v4.14.3's
  `C:tmpv4_14_1_changelog.md`).

### Tests + CI

* **2192 pass / 5 skip / 1 xfail = 2198 collected** (up from 2113
  collected at v4.16.0; +85 net across the 4 agents -- A=11, B=26,
  C=20+6, D=22).  (v4.16.2 audit P2-NEW-F2-MED-2: pre-v4.16.2
  headline cited 2208/+102/2106 -- arithmetic broken; refreshed
  here to use the collected metric, which arithmetic-reconciles.)
* New test modules:
  * `tests/unit/test_v4_16_1_agent_a.py` (11 tests)
  * `tests/unit/test_v4_16_1_agent_b.py` (26 tests)
  * `tests/unit/test_v4_16_1_agent_c.py` (20 tests)
  * `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`
    (6 walker tests — the 10th meta-pin)
  * `tests/unit/test_v4_16_1_agent_d.py` (22 tests)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch site
  drifted `:2958` -> `:2974` after Agent A's Bug 1 SUM->AVG line
  additions; the v4.15.3 + v4.15.4 entries' "now at :2958" cites are
  refreshed to `:2974` (the v4.15 line-citation pin
  `TestF5ChangelogLineCitations` verifies a citation within +/-5 of
  the live site exists in CHANGELOG).

---

## [4.16.0] — 2026-05-19

**Major minor release** rolling up the entire v4.16 + v4.17 + v4.18
ROADMAP into a single release.  4 large feature buckets ship together:
the remaining 4 V4 meta-pin candidates (closing the structural
counter-measure trajectory begun in v4.15.0); multi-process atomic-
append for `storage.py` (HDF5 SWMR + filelock distributed Zarr lock);
the full optimisation framework expansion (constrained opt,
checkpoint/resume, Newton-step, multi-objective NSGA-II via pymoo);
and the glass/materials expansion (CDGM + Hikari + Sumita catalogues
+ per-glass Sellmeier validity ranges + central cache registry).
**2106 unit tests pass** (up from 1922; +184 net), 5 documented
skips (4 pymoo + 1 ZARR_MKDIR_PATCH), 1 documented xfail; **34/34
validation files pass**.

### Bucket 1 — V4 meta-pin candidates (4 walkers complete)

The audit's standing V4 recommendation from AUDIT_V4_14_2 Part 3.5
onward.  v4.15.x landed candidates V1 (cache-clears), V2 (cache↔lock),
V3 (0+0j), V4 (`_validate_grid_params`), V5 (`_check_2d_scalar_field`).
v4.16.0 lands the remaining four — completing the meta-pin coverage
of the recurring sibling-gap classes the audits identified:

* **Sentinel-aware branch propagation walker** — AST-walks
  `_get_wrapper_merit_cache` callsites for `is _ZERO_APERTURE_MASK`
  branch.  3 sites discovered, all already guarded (v4.14.1-v4.14.3
  closures are clean).  Counter-pin verifies synthetic violation
  triggers the walker.
* **Cross-backend dispatch (`_xp_of` usage) walker** — AST-walks
  field-domain public functions for hardcoded `np.*` patterns where
  `xp = _xp_of(E); xp.<...>` should dispatch.  94 candidates
  discovered; **5 inline fixes shipped** in `lumenairy/elements/elements.py`
  (`apply_zernike_aberration`, `apply_lyot_focal_plane_mask`,
  `apply_vortex_phase_mask`, `apply_apodized_pupil`, plus the
  `zernike` helper); 56 documented exemptions.
* **`dy` parameter threading walker** — every `apply_*` in
  `lumenairy.__all__` must accept `dy: Optional[float] = None` for
  anamorphic-grid support.  36 functions discovered; 26 already
  threading `dy`, 10 documented exemptions (`apply_perturbations`
  prescription input, `apply_mask` element-wise mul, polarization
  `JonesField` helpers, bundle helpers like
  `apply_thin_lens_to_beamlets`, `apply_detector` square-grid,
  `apply_dm` mirror-geometry square).
* **`__all__` symmetry walker** — every name in submodule `__all__`
  is either re-exported at the top level OR marked `_INTERNAL`
  (`_*` prefix).  752 submodule entries; 717 re-exported; **35
  documented exemptions**; **9 inline fixes** (8 backend-array-
  namespace helpers + `PYMOO_AVAILABLE` promoted to top-level
  `__all__`).

Each walker carries a fake-violation counter-pin (positive-signal
test pattern from v4.15.0 / v4.15.4).  **All 9 dispatcher meta-pins
now active and green** (cache-clears, cache↔lock, 0+0j,
validate_grid_params, check_2d_scalar_field, sentinel-propagation,
xp-dispatch, dy-threading, __all__-symmetry).  The "fix N, miss N+1"
sibling-gap meta-pattern at the public-API surface is now
structurally retired across all currently-known classes; new
classes will continue to surface and be added to the V-walker
family as identified.

### Bucket 2 — Multi-process atomic-append for `storage.py`

v4.14.3 documented single-process atomicity for `append_plane_h5`
and `_zarr_append_plane` plus a multi-process restriction.  v4.16.0
closes the multi-writer story:

* **HDF5 SWMR mode** — `append_plane_h5` gains `swmr: bool = True`
  kwarg.  When `True`, file opened with `libver='latest'`,
  `f.swmr_mode = True` after dataset creation.  Concurrent readers
  can safely follow a single writer; multiple writers are
  serialised via the sibling lock.
* **filelock-based distributed Zarr lock** — both `append_plane_h5`
  and `_zarr_append_plane` wrap the attr-write + create-array
  sequence in a `filelock.FileLock` on the sibling `<path>.lock`
  file.  Cross-process race-free; configurable
  `lock_timeout: float = 30.0` kwarg.
* **`filelock>=3.0`** added to `hdf5` and `zarr` optional-dependency
  groups in `pyproject.toml` (verified NOT a transitive dep of
  either h5py 3.16 or zarr 3.1).
* **Subprocess multi-writer tests** via `multiprocessing.get_context('spawn')`
  (Linux + Windows portable) — 4 workers × 5 planes each verifies
  20-plane final file with no data loss.
* **Single-process v4.14.3 atomicity guarantees preserved**
  bit-for-bit.

Measured overhead: ~5× slowdown 4-writer contended vs 1-writer
baseline; <5% lock overhead on large planes (≥4096²) where
HDF5/Zarr I/O dominates.

### Bucket 3 — Optimisation framework expansion (ROADMAP v4.17)

Four additions to `lumenairy.optimize`:

* **Constrained optimisation** via scipy `NonlinearConstraint`
  mapping.  New `Constraint` dataclass; `design_optimize(...,
  constraints=[Constraint(fn=..., lb=..., ub=..., label=...)])`.
  Method-compatibility validator raises a clear `ValueError` for
  non-supporting methods (L-BFGS-B / `lm` / `differential_evolution`
  / `basin_hopping` / `dual_annealing` all silently ignored
  constraints in scipy's core API — v4.16 rejects them up front
  pointing the user at SLSQP / trust-constr).  Diagnostic
  constraint-label printed in progress callback.
* **Checkpoint / resume on long `design_optimize` runs**.  Add
  `state_file: Optional[str] = None` and
  `state_save_every: int = 1` kwargs.  Persists
  `(call_count, x_best, merit_best, history)` to JSON with atomic-
  replace write (`.tmp` + `os.replace`).  On startup, if the file
  exists with matching shape, resumes from persisted `x_best`.
  Gated on `state_file` non-None so legacy callers see byte-
  identical behaviour.
* **Multi-objective Pareto via pymoo NSGA-II wrapper** —
  `lumenairy.optimize.multi_objective.design_optimize_multi_objective(...)`
  with `ParetoResult` dataclass.  pymoo is an **optional
  dependency** in the new `multi_objective` extras group (`pip
  install lumenairy[multi_objective]`).  Same opt-in pattern as
  jax/cupy/numba/h5py/zarr in the library.  Module imports
  unconditionally; only the actual function call raises
  `ImportError` with a clear install hint.  pymoo's heavier
  transitive deps (autograd, deap, cma) are deliberately NOT
  bundled into the `all` group.  4 new top-level exports:
  `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE`.
* **Hessian / Newton-step optimisation** via `method='newton'`.
  Dispatches to scipy `trust-ncg` with FD-Jacobian-of-FD-gradient
  Hessian estimator.  `UserWarning` recommends L-BFGS-B for
  `n_params > 30` (Newton's FD-Hessian cost scales as n²).

13 tests (4 constrained + 3 checkpoint + 4 multi-objective + 2
Newton); 4 of the multi-objective skip cleanly if pymoo isn't
installed.

### Bucket 4 — Glass + materials + central cache registry

Three additions (ROADMAP v4.18 items #13-#15):

* **CDGM + Hikari + Sumita Sellmeier catalogues** — 32 new
  glasses across the three major non-Western catalogues:
  - **CDGM (12)**: H-K9L, H-LAK52, H-LAK53A, H-ZK9B, H-ZF12,
    D-ZK3, D-LAK52, H-ZLAF52A, H-ZK7, H-ZF52A, F1-CDGM, F2-CDGM.
  - **Hikari (10)**: E-LASF016, E-SK16, E-LAK7, E-LAK04, E-BAK1,
    J-FK01A, J-LASF09A, J-LAK7, J-BASF7, E-F2.
  - **Sumita (10)**: K-VC78, K-LAK10, K-LASFN10, K-SK4, K-PFK90,
    K-PBK40, K-BK7, K-PSKN2, K-FK5, K-LAFN3.
  **`GLASS_REGISTRY`: 46 → 78 entries**.  Every new entry n_d
  cross-checked against the official datasheet (or
  refractiveindex.info as proxy) at the 5e-5 tolerance pin
  established by v4.14.2's S-LAH64/79 verification.  Zero
  glasses failed the cross-check.  8 of the new CDGM glasses
  also ship as bundled Sellmeier-formula-2 fallbacks for
  minimal installs without `refractiveindex`; the remaining 22
  use formula-3 (polynomial) which requires `refractiveindex`
  for minimal installs — clear `ImportError` with install hint.
* **Per-glass Sellmeier validity ranges** — new `GLASS_VALIDITY`
  table with 77 entries (one per catalogued glass).  Format
  `{name: (lambda_min, lambda_max)}` in metres.  Extrapolating
  outside the range emits `UserWarning(...)` but does NOT raise
  — extrapolation is sometimes acceptable for design-space
  exploration.  Per-glass sources cited inline in the table
  (refractiveindex.info URLs + datasheet revs).
* **Central cache registry** (`lumenairy/_cache_registry.py`)
  — new public API `register_cache_clearer(name, clear_fn)` +
  `list_registered_cache_clearers()` + `clear_all_registered_caches()`.
  Retires the lazy-import fan-out in `clear_asm_caches`.  9
  caches migrated to the registry (`asm_local`, `lg_mode_stack`,
  `lg_polynomial_items`, `zernike_basis`,
  `through_focus_scan_jax`, `propagate_system_jax`,
  `phase_retrieval_kernels`, `trace_jax`,
  `wrapper_merit_meshgrid`).  `clear_asm_caches`'s external
  contract is preserved bit-for-bit (still callable with the
  same name + signature); the internal dispatch is now
  registry-walking instead of hand-enumerated.  Structural
  counter-measure to the cache-clear "fix N, miss N+1" pattern
  the v4.14.x audits identified.

127 new tests (10 cache-registry + 109 glass + 8 validity); zero
n_d cross-check failures.

### New top-level exports (12)

* `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE` (optimisation)
* `register_cache_clearer`, `list_registered_cache_clearers`,
  `GLASS_VALIDITY` (cache registry + glass validity)
* `array_namespace`, `is_numpy_array`, `is_cupy_array`,
  `is_jax_array`, `backend_name`, `to_numpy`, `to_backend`,
  `RandomState` (backend helpers — Agent A's `__all__` symmetry
  fix promoted these)

### Optional dependencies

* New `multi_objective` extras group: `pip install
  lumenairy[multi_objective]` adds pymoo for NSGA-II Pareto.
* `hdf5` and `zarr` extras groups now include `filelock>=3.0`
  (was previously transitively missing).

### Test counts

* Pre-v4.16.0 baseline (v4.15.5): 1922 unit pass + 1 skip + 1 xfail.
* v4.16.0 additions: A=30 (4 walkers), B=15 (multi-process storage),
  C=13 (10 pass + 3 pymoo-skip; pymoo not installed in test env),
  D=127 (10 cache-registry + 109 glass + 8 validity); plus 5 inline
  xp-dispatch fixes' positive regression coverage.  Net +184
  collected, +5 documented skips (3 new pymoo + 0 already present
  ZARR mode + 1 already present).
* Final: **2106 unit pass + 5 skip + 1 xfail; 34/34 validation**.

### ROADMAP status post-v4.16.0

* **v4.16, v4.17, v4.18 — all items shipped**.  The ROADMAP's
  Current State section is refreshed to reflect this; remaining
  target sections are folded into Shipped highlights.
* **v5.0 — immediate horizon**.  Major structural release: 6
  file splits, CI gates (pytest fast-PR + ruff + mypy --strict
  incremental + `__all__` smoke), remove 8 active back-compat
  shims, shared Chebyshev helpers extraction, audit-fix test-file
  consolidation, `lumenairy/system.py` → `propagators/system.py`,
  off-axis conic in surface frame (Optiland/Zemax parity), bump
  `requires-python` to >=3.10, 3 config knobs, docs.
* **Designer GUI v3.8+** still unplanned (separate version
  stream).

### Known issues / flagged for v4.16.1

* **Bundled Sellmeier formula-3 (polynomial) evaluator** —
  Hikari, Sumita, and 4 CDGM glasses use refractiveindex.info
  formula 3.  v4.16.0's `_sellmeier_index` only supports
  formula 2.  Minimal installs without `refractiveindex` raise
  `ImportError` on these 26 entries with a clear actionable
  message.  v4.16.1 candidate: add `_polynomial_index`
  evaluator.

### Deferred to v5.0

Architectural items requiring breaking changes — see ROADMAP for
the full v5.0 catalogue.

---

## [4.15.5] — 2026-05-19

**Closes the v4.15.4 audit (`docs/audits/AUDIT_V4_15_4_2026_05_19.md`)
through P3.**  The audit found 0 P0 + 4 P1 + 6 P2 + 5 P3.  Three of
the 4 P1s closed via a **V6 dispatcher meta-pin walker refactor**:
discovery now keys off the function's **first-positional-parameter
name** (via AST inspection of `ast.arguments`) rather than a hand-
curated name-prefix list, plus the walker now **descends into class
bodies** for non-delegating methods like `DeformableMirror.apply`.
The remaining P1 cluster (2 user-facing pitfalls in v4.15.4's new OPD
plotting functions) closed via a `fan_units` kwarg + centered-RMS
metric consistency.  **1922 unit tests pass** (up from 1858; +64
net), 1 documented skip, 1 documented xfail; **34/34 validation
files pass**.

### Headline: V6 walker (first-positional-param-name discovery)

The v4.15.3/4 dispatcher meta-pin walker used a hand-curated name-
prefix filter (`apply_*` / `propagate_*` / `richards_wolf_*` /
`debye_wolf_*`).  v4.15.4 audit found 30+ public `__all__`
functions outside the filter that take 2-D `E`/`field`/`pupil`
first positional args — the meta-pattern recurred at one indirection
level higher.

v4.15.5 refactors discovery:

* **Primary filter is now AST-based first-positional-param name.**
  Walks `lumenairy.__all__`; for each public function, inspects
  `ast.arguments.args[0].arg`; if the name is in
  `_FIELD_PARAM_NAMES = frozenset({'E', 'E_in', 'field', 'pupil',
  'object_field', 'psf'})`, requires `_check_2d_scalar_field` call
  OR `_GUARD_EXEMPTIONS` entry.  Discovery is now grounded in the
  actual function signature rather than a string match.
* **Legacy name-prefix filter retained as fallback** — v4.15.4
  coverage preserved bit-for-bit; V6 only ADDS entries.
* **Class-body descent** — the walker now visits class methods
  named `apply` / `propagate` / similar.  `_DELEGATING_CLASS_METHODS`
  exemption set documents which classes legitimately delegate to
  module-level guarded functions (operator-algebra: `ThinLens.apply`,
  `FreeSpace.apply`, `CylindricalLens.apply`, `Magnify.apply`,
  `FourierTransform.apply`, `Aperture.apply`, `GaussianAperture.apply`,
  `CompositeOperator.apply`).
* **`_file_to_ast` `lru_cache`d** (P3-NEW-F1-2) — 11 sibling
  functions in `_lens_thin.py` previously triggered 11 re-parses
  of the same file; cache cuts that to 1.  Walker test wall time:
  **~1.5s → ~0.03s warm (~30× speedup)**; cache stats `hits=180,
  misses=42, currsize=42` on a full walk (81% hit rate).

Post-refactor walker discovery:  **96 top-level entry points + 3
class methods** = 99 candidates (was 72 at v4.15.4 HEAD).  Of the
96 top-level: 39 guarded (was 25), 47 exempt (unchanged), 10 newly-
flagged for v4.16+ inline guard sweep (HFPI initialisers,
decomposition helpers, low-priority `analysis/core.py` analyzers).

### P1 closures (4)

* **P1-NEW-F2-2 — `DeformableMirror.apply` 1-line guard.**  The
  module-level `apply_dm` was guarded in v4.15.4; the class method
  was a 3-line `E_in * np.exp(1j * phi)` with no `_check_2d_scalar_field`
  call.  Walker's blanket class-method exclusion (v4.15.4 docstring
  asserted methods "delegate to guarded scalar functions") was true
  for Cluster-B operator algebra, false here.  v4.15.5 closes both:
  inline guard on the method + walker descent into class bodies.
* **P1-NEW-2WAY-1 — V6 walker refactor** (covered under headline)
  + **inline guards on the 13 highest-traffic unguarded analyzers**:
  `wave_opd_2d`, `M2`, `strehl_ratio`, `beam_d4sigma`,
  `coupling_efficiency`, `compute_psf`, `compute_otf`, `compute_mtf`,
  `encircled_energy_curve`, `koehler_image`, `extended_source_image`,
  `shack_hartmann`, `rays_from_field`, `resample_field`.  34
  regression tests pin each guard via direct `MCF` / 3-D ensemble
  rejection.
* **P1-NEW-V2-1 — `plot_opd_fan` `fan_units` kwarg.**
  `raytrace.opd_fan_data` returns OPD in **waves**;
  `plot_opd_fan` expected **metres**.  The canonical pipeline
  `plot_opd_fan(*opd_fan_data(...), units='waves', wavelength=wl)`
  divided by `wavelength` a second time → silently wrong by ~6e5.
  v4.15.5 adds `fan_units: str = 'm'` (default `'m'` preserves
  v4.15.4 callers; pass `fan_units='waves'` for the
  `opd_fan_data` pipeline).  End-to-end regression test pins
  `opd_fan_data → plot_opd_fan(fan_units='waves', units='waves',
  wavelength=wl)` does not double-convert.
* **P1-NEW-V2-2 — `_radial_rms_profile` centered RMS.**  v4.15.4
  `_radial_rms_profile` used `sqrt(mean(opd²))` (variance about
  zero) while the 1-D fan RMS and 2-D heatmap RMS used
  `sqrt(mean((opd - mean)²))` (centered, wavefront-error
  convention).  On a pure-defocus OPD, the radial-RMS curve was
  piston-dominated and looked like r²; the heatmap annotation was
  the much smaller centered RMS.  Numbers on the same figure did
  not reconcile.  v4.15.5 switches to centered RMS using the in-
  aperture mean computed once for the entire OPD.  Example
  `plot_opd_summary_singlet.py` RMS now reports **0.8901 waves**
  (was 1.3347 waves uncentered); PV unchanged at 2.9318 waves.

### P2 closures (6)

* `_check_2d_scalar_field` parameterized with `input_kind:
  str = 'field'` — `richards_wolf_focus` / `debye_wolf_psf` /
  `compute_psf` / `compute_otf` / `compute_mtf` etc. pass
  `input_kind='pupil'` or `'psf'`, getting accurate error
  messages ("expected 2-D complex pupil" instead of "field").
* `plot_opd_summary` docstring corrected to explicitly state
  `opd_2d` input is in **metres** (matching `plot_wavefront`'s
  convention).
* Example `plot_opd_summary_singlet.py` RMS print switched to
  centered form (matches heatmap annotation).
* `plot_opd_summary` even-N central-row/col fallback: aligned
  using `(N - 1) // 2` consistently.
* Walker descends into class bodies (covered under headline).
* CHANGELOG v4.15.4 entry-point count refresh (43→49 → actual
  72 at v4.15.4 HEAD; v4.15.5 numbers 96 + 3 class methods are
  documented in this entry).

### P3 closures (5)

* `_file_to_ast` `@lru_cache` (covered under headline).
* `plot_opd_fan` docstring now explicitly states "centered RMS
  (about the in-aperture mean); PV (max - min)".
* `n_bins` kwarg exposed on `plot_opd_summary` as
  `radial_rms_n_bins: int | str = 'auto'` with auto-clamping for
  tiny grids (`min(32, int(sqrt(N_in_aperture)) // 2)`).
* CHANGELOG `-W error::DeprecationWarning` failure count drift
  (57 vs 63) reconciled — canonical at v4.15.4 commit time was
  63; v4.15.3 audit's 57 reflects pre-dispatch count, drift
  documented inline.
* CHANGELOG per-agent attribution arithmetic +1 footnote added
  (per the standard pattern from v4.15.3 hygiene fix).
* CHANGELOG wavelength annotation drift fixed (587.56 nm → 633
  nm to match the example).
* ROADMAP duplicate-counting drift cleaned up: 6 already-shipped
  items moved from "v4.16 residual" to "Shipped highlights"
  (polychromatic encircled energy, polarisation-aware Strehl,
  resolution metrics, astigmatism mag+angle, OAP, Forbes Q-type
  — all landed in v4.15.0 / v4.15.1).  True v4.16 residual is
  now 2 items: V4 meta-pin candidates + multi-process atomic-
  append for `storage.py`.  Remaining v4.17/v4.18 items
  renumbered.

### Test counts

* Pre-v4.15.5 baseline (v4.15.4): 1858 unit pass + 1 skip + 1 xfail.
* v4.15.5 additions: A=34 regression + 4 V6/class-method pins
  (38 total), B=17 + 3 carry-forward (20 total), C=4.  Net +64.
* Final: **1922 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker; the V6
first-positional-param-name candidate landed in v4.15.5);
MCF-aware downstream propagators; multi-process atomic-append
for `storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.  Plus: **10 newly-flagged
unguarded analyzers** for v4.16+ inline-guard sweep (HFPI
initialisers, decomposition helpers, lower-priority
`analysis/core.py` analyzers like `beam_centroid`,
`beam_diameter`, `radial_power_bands`, `wave_opd_1d`,
`strehl_phase_integral`, resolution metrics, `single_plane_metrics`).

---

## [4.15.4] — 2026-05-19

**Closes the v4.15.3 audit (`docs/audits/AUDIT_V4_15_3_2026_05_18.md`)
through P3 + adds two user-facing OPD plotting functions.**  The audit
found 0 P0 + 1 P1 + 6 P2 + 5 P3 — the cleanest yield in the v4.15.x
series.  The single P1 is the recurring "fix N, miss N+1" meta-
pattern re-emerging one level of indirection higher than v4.15.3
closed it: the dispatcher meta-pin's `_TARGET_PACKAGES` scope itself
was a sibling gap.  **1858 unit tests pass** (up from 1822; +36 net),
1 documented skip, 1 documented xfail; **34/34 validation files pass**.

### Headline: walker scope refactor closes the meta-pattern at the package level

v4.15.3 shipped the `_check_2d_scalar_field` helper + dispatcher meta-
pin.  But the walker scoped only to
`('lumenairy/propagators', 'lumenairy/elements')`, missing 4 public
entry points outside that scope.  v4.15.4 makes discovery
`__all__`-based:

* **`_walk_entry_points` refactored** to walk `lumenairy.__all__`
  membership via `inspect.getsourcefile`; survives future refactors
  that move functions between subpackages.  Package-walk retained as
  a fallback.  Walker discovery at v4.15.4 HEAD: **72 total entry
  points (25 guarded + 47 documented exempt)**.  (The pre-correction
  CHANGELOG bullet cited "43 -> 49 after the refactor", which
  reflected the `__all__`-pass-only count without the package-walk
  fallback dedup; v4.15.5 / Agent C refreshed this from the live
  diagnostic per AUDIT_V4_15_4 P2-NEW-3WAY-3.  TODO: the v4.15.5 V6
  walker refactor (Agent A scope) will change this number again at
  integration time; Agent C populated this with v4.15.4 HEAD numbers
  and integration will update with v4.15.5 V6-refactored numbers.)
* **Name filter broadened** with `name.startswith('propagate_')` to
  catch `propagate_through_system` + `propagate_through_system_jax`
  (which contain `propagate_` at the start but not `_propagate` in
  the middle — the v4.15.3 filter missed both).
* **6 newly-found sibling entry points guarded** via the v4.15.3
  helper: `propagate_through_system_jax` (P1), `apply_dm`,
  `apply_detector`, `richards_wolf_focus`, `debye_wolf_psf`; plus
  `apply_perturbations` documented exempt (first positional arg is
  a prescription dict, not a 2-D scalar field).
* **Fake-violation counter-pin** added to the dispatcher meta-pin:
  injecting a synthetic unguarded function via `monkeypatch.setattr`
  must trigger the meta-pin's `AssertionError`.  Walker correctness
  now pins on a positive signal.

### P1 closure (1)

* **P1-NEW-3WAY-1** walker scope completeness — closed via
  `__all__`-based discovery + extended `_TARGET_PACKAGES`.

### P2 closures (6)

* Walker name-regex broadening + 3 unguarded `analysis/` siblings
  guarded (covered under headline).
* SAS-anamorphic CHANGELOG wording corrected retroactively in the
  v4.15.3 entry: `"forces method='asm' regardless of self.method"`
  -> `"forces method='asm' when self.method == 'auto' and dy != dx;
  explicit method='sas' on anamorphic grids still crashes (user's
  responsibility)"`.
* `_validation.py` lazy-import hoisted to module scope.  The
  v4.15.3 code used a lazy import citing a hypothetical circular
  dep; audit grep-verified no actual circular dep exists.  Saves
  ~1 µs/call (1-10 ms per merit eval in optimization loops with
  thousands of propagator calls).
* Dead `_PerturbedABCDFallbackSentinel` deleted.  v4.15.3 marked
  it dead via `_v4_15_3_dead_code = True` class attribute
  (informational only — no static analyzer honors it).  v4.15.4
  deletes the class + singleton (~58 LOC).  v4.15.2 test pin
  updated in the same commit.
* ROADMAP refreshed: post-v4.15.3 test count ~1750 → actual 1824;
  AUDIT_V4_15_2 + AUDIT_V4_15_3 added to closed-audits list; meta-
  pin coverage 3 of 5 → 4 of 5 with the V5 entry describing
  `_check_2d_scalar_field`.

### P3 closures (5)

* CHANGELOG dispatcher meta-pin count drift fixed (18/25 → 17/26;
  43 total).
* Fake-violation counter-pin added (covered above).
* CHANGELOG stacklevel wording: "6 Source classmethod shims" → "5
  `Source.*` classmethod shims at `:2424, 2510, 2587, 2661, 2750`
  plus the module-level `create_led_source` factory shim at
  `:1209`".
* `-W error::DeprecationWarning` test hygiene: 63 v4.15.3 tests
  previously failed under the strict flag (they exercised the
  documented Schell `return_kind` default-path warning without
  shielding).  v4.15.4 adds `pytestmark =
  pytest.mark.filterwarnings('default::DeprecationWarning')` to 6
  affected test files.  Failures: 63 → 0.  *(Note: the v4.15.3
  audit's P3-NEW-F2-4 cited 57 failing tests; the discrepancy with
  this CHANGELOG's 63 reflects pre/post-v4.15.4 dispatch-test-file
  additions between when the audit was filed and v4.15.4 commit
  time.  Canonical count at v4.15.4 commit time was 63; closure
  verified to 0 escalations regardless of which baseline is right.
  Documented per AUDIT_V4_15_4 P3-NEW-V3-1 / v4.15.5 Agent C.)*
* CHANGELOG test-count arithmetic reconciled: v4.15.3 baseline
  1732 → 1733; per-agent attribution sum 88 documented alongside
  actual collected delta 89 with explicit explanation of the +1
  attribution-vs-collection gap.  Removed the false claim about a
  ROADMAP update that wasn't actually performed in v4.15.3.

### New: OPD plotting functions

Two new public functions in `lumenairy/analysis/plotting.py`, visually
matching the `OPDPy_Lumenairy_Crosscheck` `fig_variety_L*.png` style:

* **`plot_opd_fan(py, opd_y, px, opd_x, *, wavelength=None,
  units='waves', show_stats=True, title=None, fig=None, axes=None)`**
  — 2-panel tangential + sagittal OPD fans.  Inputs match
  `lumenairy.raytrace.opd_fan_data`'s return tuple.  Solid-line
  plots with zero-reference axhline, in-axes PV/RMS annotation,
  units kwarg matches `plot_wavefront` (waves / nm / um / m).
  Returns `(fig, (ax_y, ax_x))`.  147 LOC.
* **`plot_opd_summary(opd_2d, dx, *, dy=None, aperture=None,
  wavelength=None, py=None, opd_y=None, px=None, opd_x=None,
  units='waves', cmap='RdBu_r', show_stats=True, title=None,
  fig=None)`** — 4-panel summary: 2-D heatmap (delegates to
  `plot_wavefront`), radial RMS profile (32 annular bins),
  tangential fan, sagittal fan.  Fan panels use the provided
  `(py, opd_y, px, opd_x)` if supplied (preferred — raytrace data
  has chief-ray reference built in) or auto-extract from the 2-D
  OPD's central row/column otherwise.  Returns `(fig, ((ax_hm,
  ax_rms), (ax_y, ax_x)))`.  204 LOC.

Both added to `lumenairy.__all__` (analysis tier).  10 unit tests +
runnable example at `examples/plot_opd_summary_singlet.py` (singlet
OPD via `apply_real_lens_traced` + `opd_fan_data`; PV ≈ 2.93 / RMS
≈ 1.33 waves at λ=633 nm).  *(Pre-v4.15.5 wording cited
λ=587.56 nm, contradicting the example's actual ``wavelength =
633e-9``; corrected per AUDIT_V4_15_4 / v4.15.5 Agent C.)*

### Test counts

* Pre-v4.15.4 baseline (v4.15.3): 1822 unit pass + 1 skip + 1 xfail.
* v4.15.4 additions, per-agent attribution: A=8 + 1 counter-pin in
  the meta-pin file (=9), B=11, C=7, D=10.  Per-agent sum: **37**.
* Actual `pytest --collect-only` delta: 1858 - 1822 = **+36**
  (canonical post-release number).  The +1 attribution-vs-collection
  gap reflects the standard parametrize/fixture artifact (one of
  the new tests expands to 2 collected items via `parametrize`, or
  a fixture-only addition isn't cleanly attributed to a single
  agent).  Same pattern documented in the v4.15.3 entry; pinned by
  `test_changelog_per_agent_breakdown_sums_to_net_delta` in
  `test_v4_15_4_agent_c.py`.  Documented per AUDIT_V4_15_4
  P3-NEW-V3-2 / v4.15.5 Agent C.
* Final: **1858 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker); MCF-aware
downstream propagators; multi-process atomic-append for
`storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.

---

## [4.15.3] — 2026-05-18

**Closes the v4.15.2 audit (`docs/audits/AUDIT_V4_15_2_2026_05_18.md`)
through P3.**  The audit identified 1 P0 + 4 P1 + ~6 P2 + ~4 P3 —
mostly the recurring "fix N, miss N+1" sibling-gap meta-pattern that
has appeared in every audit round from v4.13.x onward.  v4.15.3
closes the P0 with a **structural counter-measure** (shared
validation helper + dispatcher meta-pin) that makes the recurrence
impossible going forward.  **1822 unit tests pass** (up from 1733;
+89 collected vs v4.15.2 HEAD), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Headline: structural counter-measure for the sibling-gap meta-pattern

The v4.15.2 closure guarded 10 propagator/lens entry points against
`PartialCoherenceMCF` + 3-D ensemble inputs.  This audit found **9
more public entry points** of the same type that were missed —
`angular_spectrum_propagate_tilted`, `*_propagate_mft` (3 variants),
`apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
`apply_axicon`, `apply_real_lens_traced`, `apply_real_lens_maslov`.

v4.15.3 fixes this **structurally**:

1. **`lumenairy/_validation.py`** (NEW) — single canonical
   `_check_2d_scalar_field(E, fn_name)` helper consolidates the
   `PartialCoherenceMCF` and `ndim != 2` guards.  Replaces ~240 LOC
   of duplicated boilerplate across 10 v4.15.2 sites + 9 new sibling
   sites.  Net `lumenairy/` LOC change: roughly +23 (helper +102 LOC;
   migrated sites -160 LOC; new sibling guards +81 LOC) vs +225 LOC
   the inline pattern would have cost on the 9 new sites alone.

2. **Dispatcher meta-pin**
   (`tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py`) —
   AST-walks every `def apply_*` and `def *_propagate*` in
   `lumenairy/propagators/` and `lumenairy/elements/`; asserts
   `_check_2d_scalar_field` is the first executable statement of
   each function body.  **43 entry points discovered, 17 guarded,
   26 documented exemptions** (GBD beamlets, HFPI/HF state objects,
   batched 3-D variants, JAX-traceable lens kernels, polarization
   helpers, etc.).  Adding a new public entry point in the at-risk
   modules WITHOUT the helper call now fails CI.

3. **`_GUARD_EXEMPTIONS` registry** documents every legitimate
   exemption with a reason — converts "easy to miss" into "easy to
   see in code review".

This is the 5th structural meta-pin in the library:  v4.14.1
cache-clear dispatcher pin, v4.14.2 cache↔lock pairing + 0+0j
literal sweep, v4.15.0 `_validate_grid_params` input-validation
entry-point pin, v4.15.3 `_check_2d_scalar_field` pin.

### P0 closure

**P0-NEW-F2-1 — 9 unguarded propagator/lens entry points.**  All 9
sibling sites now call `_check_2d_scalar_field` as the first
executable statement, identical guard semantics to the 10 v4.15.2
sites:
* `angular_spectrum_propagate_tilted`, `angular_spectrum_propagate_mft`,
  `fresnel_propagate_mft`, `fraunhofer_propagate_mft` in
  `propagators/propagation.py`
* `apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
  `apply_axicon` in `elements/_lens_thin.py`
* `apply_real_lens_traced` in `elements/_lens_traced.py`
* `apply_real_lens_maslov` in `elements/lenses_maslov.py`

30 regression tests pin the 9 new guards (TypeError on `PartialCoherenceMCF`,
ValueError on 3-D ensemble).  7 meta-pin tests pin the structural
counter-measure (walker discovery, helper-is-first invariant,
counter-pin against accidentally-removed guards).

### P1 closures (4)

* **P1-NEW-F1-1 — `FreeSpace._apply` SAS-anamorphic crash fixed.**
  When `dy != dx` and `method='auto'`, the dispatcher routed to SAS
  (square-grid-only); the v4.15.2 dy-threading fix passed `dy` to
  SAS which doesn't accept it.  v4.15.3 forces `method='asm'` when
  `self.method == 'auto'` and `dy != dx` — `auto` is now a hint,
  not a contract.  Explicit `method='sas'` on anamorphic grids
  still crashes (user's responsibility — the in-code comment at
  `algebra/primitives.py:142-147` documents this gating).
  `FourierTransform._apply` inherits the fix by composition (the
  3-stage rewrite creates `FreeSpace` instances internally).
* **P1-NEW-F1-2 — Schell `DeprecationWarning` stacklevel fixed.**
  `_warn_schell_return_kind_default` had `stacklevel=4`; the call
  chain is 5 frames deep (warnings.warn → warn_deprecated_signature
  → _warn_schell_return_kind_default → factory → user).  Bumped to
  5.  Library-wide sweep of `_warn_*` helpers found 6 additional
  off-by-one stacklevels in `sources/core.py` (5 `Source.*`
  classmethod shims at `:2424, 2510, 2587, 2661, 2750` plus the
  module-level `create_led_source` factory shim at `:1209`); all
  bumped 3 → 4.
* **P1-NEW-F1-3 — 3 dead `optimize/core.py` sentinels wired.**
  v4.15.2 added `_InvalidFocalLengthSentinel`,
  `_FailedScanStrehlSentinel`, `_PerturbedABCDFallbackSentinel`
  class definitions but never wired them at callsites.  v4.15.3
  wires the 2 scalar sentinels at `optimize/core.py:2424, 2696,
  3015` (was raw `-1.0` / `float('nan')` / `0.0` returns); marks
  `_PerturbedABCDFallbackSentinel` as dead-code (tuple shape didn't
  sentinel cleanly without breaking downstream unpacking; class
  retained with `_v4_15_3_dead_code = True` marker for v4.15.2
  test-pin compatibility).
* **P1-NEW-F1-4 — `Source.gaussian_schell`/`schell_model`
  classmethods route through sentinel.**  Pre-v4.15.3 these
  classmethods hardcoded `return_kind='ensemble'`, bypassing the
  v4.15.2 `_RETURN_KIND_UNSET` sentinel — calling them without
  `return_kind` produced a silent 4-tuple with no
  `DeprecationWarning` and a `Source` whose `E.ndim == 3` (every
  other `Source.*` produces 2-D).  v4.15.3 routes both
  classmethods through the sentinel; default-path callers now
  get the same DeprecationWarning as the top-level factories.
  Soft 2-D `Source.E` invariant break documented in classmethod
  docstrings as intentional (Schell is partial-coherence;
  collapsing to 2-D would be physically wrong).

### P2 closures

* **`_RETURN_KIND_UNSET` promoted** from bare `_Sentinel` instance
  to dedicated `_SchellReturnKindUnsetSentinel(_Sentinel)`
  subclass with `_SENTINEL_REGISTRY` entry for pickle round-trip
  safety.  Consistent with `_ZeroApertureMaskSentinel`,
  `_AngleUnsetSentinel`, `_NoDefaultSentinel`.
* **rays_from_field threshold-comparison consistency.**  Audit
  finding was inverted (`_place_rejection` was already `>=`,
  `_place_uniform` and `_place_cdf` were strict `>`).  v4.15.3
  makes all 3 modes inclusive `>=`.  Pixels at exactly
  `intensity_threshold` now consistently survive.
* **Non-tautological FourierTransform pin** added — Gaussian-beam
  waist relation `w_out = lambda * f / (pi * w_in)` (Saleh &
  Teich §3.2.2) through `FourierTransform(f)`.  Measured error
  <0.0001% vs the 5% tolerance pin (50000× headroom).  Pins
  physics not implementation.
* **4-fold mirror folded-prescription test cases** added to
  `test_v4_15_1_agent_g_matches_system_abcd.py` —
  `_build_folded_4fold_periscope` and
  `_build_folded_cassegrain_2curved_2flat` strengthen the
  `from_prescription` flat-mirror parity claim across more
  complex folded geometries.
* **Library-wide `_warn_*` stacklevel sweep** (~12 helpers
  audited; 7 adjusted, 5 unchanged with documented rationale).

### P3 closures

* **CHANGELOG Forbes Q OPD bullet corrected**: tolerance
  `1e-3` → `5e-3` (test code was always `5e-3`); formula
  `OPD = -k * sag` → `OPD(r) = (n - 1) * sag(r)` (the `(n - 1)`
  index factor was missing).  Test code was always correct;
  only the CHANGELOG bullet lied.
* **CHANGELOG sentinel-migration line citations refreshed**
  after Agent C's v4.15.3 wiring drift: `_ZERO_APERTURE_MASK`
  branch now at `optimize/wrapper_merits.py:855` (was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split, which
  moved `ToleranceAwareMerit.evaluate` out of the monolithic
  `optimize/core.py`; was `:3015` pre-v4.16.3 Agent C `Constraint`
  auto-probe DeprecationWarning latch + pickle catch widening; was
  `:2974` pre-v4.16.2 Agent B `MultiWavelengthMerit` `FutureWarning`
  latch + Constraint probe move + lambda pickle-probe; was `:2958` pre-v4.16.1 Agent A
  `MultiWavelengthMerit` `SUM`->`AVG` refactor; was `:2980` pre-
  v4.15.4 Agent B `_PerturbedABCDFallbackSentinel` deletion; was
  `:2905` in the v4.15.2 entry).
* **Test count reconciliation**:
  `pytest --collect-only` → 1735 collected at v4.15.2 HEAD
  (was reported as "1732 pass + 1 skip + 1 xfail = 1734" in
  CHANGELOG, off by 1); reconciled in this entry's `### Test
  counts` block below.  The ROADMAP refresh originally claimed in
  this bullet ("~1700 → 1822 baseline") did NOT land in v4.15.3
  — that drift is documented in `AUDIT_V4_15_3` P2-NEW-V3-3 and is
  closed by v4.15.4 (Agent C scope).

### Test counts

* Pre-v4.15.3 baseline (v4.15.2): 1733 unit pass + 1 skip + 1 xfail
  = 1735 collected (per the corrected v4.15.2 entry's
  `pytest --collect-only` reconciliation; pre-v4.15.3 the v4.15.3
  block transcribed this baseline as "1732" — a one-off carry-over
  of the same off-by-one the v4.15.2 entry self-corrected).
* v4.15.3 additions, per-agent attribution: A=37 (7 meta-pin + 30
  regression), B=24, C=19 (12 new file + 4 4-fold + 3
  Gaussian-waist), D=8 (7 doc + 1 boundary-regression).  Per-agent
  sum: **88** (pre-v4.15.4 corrected from "Net +90" — neither the
  per-agent sum nor the collected delta were ever 90).
* Actual `pytest --collect-only` delta against the v4.15.2 baseline
  (1735) at v4.15.3 HEAD sha `7808107`: **+89 collected** (1824
  collected at v4.15.3 HEAD); the +1 gap between the per-agent
  attribution sum (88) and the collected delta (89) is a
  parametrize-expansion / fixture-collection artifact that does
  not cleanly attribute to a single agent; the canonical number is
  the collected delta.
* Final: **1822 unit pass + 1 skip + 1 xfail = 1824 collected**;
  **34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates (sentinel-aware
branch propagation, `_xp_of` dispatch, `dy` parameter threading
walker including ThinLens + lens kernels, `__all__` symmetry
walker); MCF-aware downstream propagators; multi-process
atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor`; Forbes Q-2D-
asymmetric variant.  Plus newly-deferred: 9 `elements/elements.py`
generic helpers + 2 JAX-traceable lens variants + 6 polarization
helpers exempted in v4.15.3 meta-pin pending v4.16 integration.

---

## [4.15.2] — 2026-05-18

**Closes the v4.15.1 audit (`docs/audits/AUDIT_V4_15_1_2026_05_18.md`)
through P3.**  The audit found 1 P0 + 9 P1 + ~12 P2 + ~10 P3 — most
were downstream-integration gaps from the rapid v4.15.1 expansion (new
types shipped without updating consumers; breaking changes shipped
without CHANGELOG flagging; primitive APIs asymmetric).  **1732 unit
tests pass** (up from 1625; +107 net), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Breaking changes

* **Schell-family factories emit `DeprecationWarning`** on the default
  `return_kind` path.  v4.15.1 silently changed the return shape from
  `(E_2d, x, y)` (v4.15.0) to `(E_3d, dx, dy, wavelength)` 4-tuple
  without a warning.  v4.15.2 closes the contract break: callers must
  pass `return_kind='ensemble'` or `return_kind='mcf'` explicitly;
  failing to do so emits a one-release deprecation warning with
  `version_removed='5.0'`.
* **`Source.gaussian_schell` and `Source.schell_model` classmethods**
  now return the same `(ensemble, dx, dy, wavelength)` 4-tuple as the
  top-level factories — they previously wrapped the 3-D ensemble in a
  `Source` instance whose `E` was 3-D, breaking the canonical 2-D
  `Source.E` contract.  This is a soft consistency break (every other
  `Source.*` classmethod returns a `Source`); the inconsistency is
  honest — Schell is partial-coherence, fundamentally different from
  the coherent single-source abstraction.

### P0 closure

**P0-NEW-1 — Schell silent contract break closed.**  `_RETURN_KIND_UNSET`
module-level sentinel (subclass of `_deprecation._Sentinel`) detects
the default-path entry; `_warn_schell_return_kind_default` helper
routes through `_deprecation.warn_deprecated_signature` with explicit
`version_removed='5.0'`.  Applied to all 3 Schell factories
(`create_gaussian_schell_source`, `create_schell_model_source`,
`create_annular_incoherent_source`).

### P1 closures (9)

* **P1-NEW-A — `FourierTransform` 3-stage rewrite.**  v4.15.1's
  `_apply` ran 2 stages (lens-then-Fresnel) while the ABCD claim
  `[[0, f], [-1/f, 0]]` matched the 3-stage chain `FreeSpace(f) *
  ThinLens(f) * FreeSpace(f)`.  The 2-stage path left a residual
  `exp(+ik/(2f) r^2)` quadratic phase — ABCDs matched but fields
  didn't.  v4.15.2 rewrites `_apply` to the literal 3-stage chain so
  ABCD and field finally agree.  Perf impact: ~2x slower than the
  v4.15.1 2-stage shortcut (one extra Fresnel propagation).  Users
  wanting hardware-realistic back-focal-plane semantics can compose
  directly as `ThinLens(f) * FreeSpace(f)` (which has the genuine
  2-stage ABCD `[[1, f], [-1/f, 0]]` and 2-stage field; both correct
  via the existing algebra).
* **P1-NEW-B — `from_prescription` flat-mirror parity matches
  `system_abcd`.**  v4.15.1 flipped `mirror_parity` unconditionally
  on `is_mirror=True`; `system_abcd` only flips for curved mirrors.
  v4.15.2 conditions the parity flip on curved mirrors, matching the
  raytrace convention.  New folded-singlet + folded-telephoto
  prescription test cases pin the parity at 1e-12 absolute (the
  1e-12 ABCD parity claim now holds for folded prescriptions too,
  not just non-folded).
* **P1-NEW-C — `FreeSpace._apply` threads `dy`.**  v4.15.1's
  `FreeSpace._apply` called `propagate(E, z=, ..., dx=dx, method=)`
  without `dy`.  Any anamorphic chain `Magnify(a_x, a_y) *
  FreeSpace(d)` silently propagated on the wrong grid.  v4.15.2
  threads `dy` to the dispatcher when `dy != dx` (forwards via
  method_kwargs; safe for ASM/Fresnel/Fraunhofer/RS; skipped for
  SAS which is square-grid only).  Verified `ThinLens._apply` does
  not have the same gap.
* **P1-NEW-D — `rays_from_field` `'cdf'` placement pixel-wise
  threshold.**  v4.15.1 applied `intensity_threshold` to MARGINAL
  sums in `_place_cdf`, inconsistent with `'rejection'` and
  `'uniform'` modes (which threshold pixel-wise).  A 1-pixel-wide
  bright streak running the full y-extent survived the threshold
  incorrectly.  v4.15.2 thresholds pixel-wise before forming
  marginals; the 3 placement modes are now consistent.
* **P1-NEW-E — `PartialCoherenceMCF` defensive guard.**  All 10
  propagator entry points (`propagate_through_system`, `propagate`,
  `angular_spectrum_propagate`, `fresnel_propagate`,
  `fraunhofer_propagate`, `rayleigh_sommerfeld_propagate`,
  `scalable_angular_spectrum_propagate`, `apply_thin_lens`,
  `apply_cylindrical_lens`, `apply_real_lens`) now raise
  `TypeError` with a clear "v4.16+ scope" message when handed a
  `PartialCoherenceMCF` — previously crashed with cryptic
  `AttributeError`.
* **P1-NEW-F — 3-D ensemble shape guard.**  Same 10 propagator
  entry points now raise `ValueError` on `E.ndim != 2` with a
  message showing the iterate-over-ensemble workaround pattern.
* **P1-NEW-G — CHANGELOG `### Breaking changes` subhead** added
  to the v4.15.1 entry listing Schell return shape, `strehl_vector`
  default-reference removal, and `system.evaluate` mixed-shape
  `ValueError`.
* **P1-NEW-H — `rays_from_field` short-return `RuntimeWarning`.**
  v4.15.1's `_place_rejection` and `_place_uniform` could return
  fewer rays than requested (rejection budget exhausted; threshold
  excluded too many pixels) without warning.  v4.15.2 emits a
  `RuntimeWarning` when `n_actual < n_rays`, plus an `n_rays = 0`
  early-return is honoured cleanly.
* **P1-NEW-I — ROADMAP refresh.**  Header bumped to "(post-v4.15.2)";
  Current State block updated to v4.15.2 / 1732 tests baseline;
  v4.15.1 + v4.15.2 added to Shipped highlights.

### P2 closures

* `_sentinel_unpickle` fallback now raises `ImportError` with an
  actionable message when an unknown subclass is unpickled
  (distributed-pipeline timing safety).  Previously silently
  returned a base `_Sentinel`, losing subclass identity.
* **3 additional `optimize/core.py` sentinels migrated** to inherit
  from `_deprecation._Sentinel`: `_InvalidFocalLengthSentinel`
  (was a literal `1e9` fallback for failed ABCD), `_FailedScanStrehlSentinel`
  (was `0.0`), `_PerturbedABCDFallbackSentinel` (was a `(efl, bfl)`
  tuple fallback).  v5.1.0 (Wave-4 integration / Agent E 6-file
  split): class definitions moved to `optimize/context.py:112-139`
  (the 2 remaining sentinels post-v4.15.4
  `_PerturbedABCDFallbackSentinel` deletion); was at
  `optimize/core.py:2069`, `:2096`, `:2122` (singletons at `:2093`,
  `:2119`, `:2144`) within the `:2044-2144` documentation block
  pre-v5.1.0.
  All registered in `_SENTINEL_REGISTRY` for pickle round-trip
  safety.  v4.15.3 correction (per AUDIT_V4_15_2 P3 docs-drift
  finding): the pre-v4.15.3 release notes cited stale work-in-
  progress line numbers `:2151, :2271, :2530, :2772` for these
  classes; the actual definitions are at the `:2044-2144` block
  cited above.  (Callsite migration to the new sentinels is
  scaffolding-only at v4.15.2 -- the `2271`, `2530`, `2772`
  references appearing inside the class docstrings point at the
  v4.16+ migration target callsites and are tracked separately by
  Agent A's v4.15.3 work.)
* `_NO_DEFAULT` promoted to dedicated `_NoDefaultSentinel(_Sentinel)`
  subclass (cosmetic consistency with the other sentinels).
* **`PartialCoherenceMCF.coherence_at` Hermiticity test** added —
  asserts `J(r1, r2) == conj(J(r2, r1))` for several `(r1, r2)`
  pairs at 1e-10.
* **`Source.gaussian_schell` / `Source.schell_model`** now pass the
  factory's 4-tuple verbatim instead of wrapping the 3-D ensemble
  in a `Source` whose `E` would have been 3-D.
* **UI runtime test under `-W error::DeprecationWarning`** added —
  exercises `SourceDefinition.to_source()` at runtime to catch the
  static-grep escape (which the v4.15.1 audit identified as a
  missing coverage class).
* **`rays_from_field` top-of-file docstring** corrected from
  Madelung `Im(grad E / E)` to phase-ratio central difference
  (inline docstring was already correct; top was stale).
* **`Magnify` docstring direction inverted** to match code:
  `a > 1` shrinks output; `0 < a < 1` magnifies (Nazarathy/Shamir
  `V[a]` convention).  Dead `operators.py:556-577` reference
  removed.
* **`'uniform'` and `'unwrap_gradient'` modes** in `rays_from_field`
  now have direct test coverage (audit flagged these as previously
  untested).  Vortex direction-recovery and anamorphic
  direction-cosines tests added.

### P3 closures

* **Sparrow tolerance pin tightened from 5% to 1%**.  Measured
  achievable error on canonical Airy fixture (N=256, dx=0.1µm,
  λ=600nm, f/#=4): 0.017% — comfortable headroom over the new 1%
  pin.  Docstring "Accuracy (v4.15.2)" paragraph cites the measured
  number.
* **Forbes Q-bfs end-to-end OPD analytical pin** — closes the
  v4.15.1 audit gap "No end-to-end Forbes Q OPD pin against
  analytical formula".  Pins `OPD(r) = (n - 1) * sag(r)` (the
  `(n - 1)` index factor reflects that light experiences an optical
  path difference of `(n_glass - n_outside) * geometric_path`; for a
  sag in vacuum/air the multiplier is `(n_glass - 1)`) against the
  closed-form Q-bfs sag at 5e-3 rad tolerance.  v4.15.3 correction
  (per AUDIT_V4_15_2 P3 docs-drift finding): pre-v4.15.3 the bullet
  stated `phi(r) = -k * sag(r)` at `1e-3 rad` -- the formula omitted
  the `(n - 1)` factor that the actual test code uses, and the
  tolerance was incorrectly tightened from the test's `5e-3` value
  (the test code itself was always correct; only the CHANGELOG bullet
  drifted).
* **`lumenairy.algebra` exports moved from Tier-2 to Tier-1** in
  `__init__.py.__all__` — operator algebra is a build-time
  construction surface, not a propagation surface.
* **CHANGELOG line-citation refreshes**: "45° fold" → "60° fold
  (α=π/6)" in the v4.15.1 OAP raytrace test description;
  `optimize/core.py:2790-2795` → `:2905` (branch) + `:2034`
  (class) + `:2044` (singleton) after Agent E's sentinel
  refactor pushed lines.
* **CHANGELOG test-count arithmetic refresh**: Agent A (v4.15.1)
  count corrected 18 → 19; Agent F count corrected 13 → 20
  (parametrize entries) to match `pytest --collect-only`.
* **`energy_threshold` kwarg** now forwarded through all 3 Schell
  factories to `PartialCoherenceMCF.from_ensemble` (was exposed on
  the MCF builder but not on the factory entry points).
* **Stray `C:tmpsources_diff.txt`** (typo'd `C:\tmp\` path; OneDrive
  U+F03A colon substitute) deleted — 44 KB git-diff dump, content
  recoverable via git history.

### Test counts

* Pre-v4.15.2 baseline (v4.15.1): 1625 unit tests + 1 skip + 1 xfail.
* v4.15.2 additions: A=18, B=15 (9 new + 6 modifications), C=32,
  D=19, E=16; net +110 (pytest-collected delta from 1625 baseline
  to 1735 at v4.15.2 HEAD sha `672051c`).
* Final: **1733 unit tests passing, 1 skipped, 1 xfailed** (1735
  collected total per `pytest --collect-only -q tests/unit` at sha
  `672051c`); **34/34 validation files passing**.  v4.15.3
  correction (per AUDIT_V4_15_2 P2 docs-drift finding): pre-v4.15.3
  this entry stated "1732 pass + 1 skip + 1 xfail" (= 1734
  collected), off by 1 from the actual `pytest --collect-only`
  count at the v4.15.2 release commit.  The arithmetic was also
  inconsistent with the per-agent breakdown above (18 + 15 + 32 +
  19 + 16 = 100, not 107) -- v4.15.3 reconciles both to the actual
  pytest-collected delta.

### Deferred to v4.16+

Unchanged from v4.15.1 deferrals: modal-asymptotic independent
ground-truth pin against direct quadrature; 4 V2 meta-pin candidates
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy` parameter
threading walker, `__all__` symmetry walker); MCF-aware downstream
propagators (consume `PartialCoherenceMCF` through propagation
chains); multi-process atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
carryover); Forbes Q-2D-asymmetric variant.

---

## [4.15.1] — 2026-05-18

**Closes the v4.15.0 audit (`docs/audits/AUDIT_V4_15_0_2026_05_18.md`)
through P3 + ships 2 additive features from
`docs/audits/CLUSTER_B_SPEC.md` (operator algebra + rays-from-field
bridge).**  The audit found 2 P0s + 12 P1s + many P2/P3 (highest-yield
audit in the series).  v4.15.1 closes both P0s + all Tier-0 P1s + the
P2/P3 sweep + adds 800+ LOC of new CLUSTER_B surface.  **1625 unit
tests pass** (up from 1425; +200 net), 1 documented skip, 1
documented xfail; **34/34 validation files pass**.

### Breaking changes

v4.15.1 ships 3 confirmed breaking items.  Callers who relied on the
v4.15.0 contracts must migrate; v4.15.2 (P1-NEW-G closure) adds this
subhead retroactively to make the audit-flagged items discoverable.

1. **Schell-family return shape**: `create_gaussian_schell_source`,
   `create_schell_model_source`, and `create_annular_incoherent_source`
   default to `return_kind='ensemble'` and now return the 4-tuple
   `(ensemble_3d, dx, dy, wavelength)` where `ensemble_3d` has shape
   `(n_realizations, Ny, Nx)`.  v4.15.0 returned `(E_2d, x, y)` (a
   collapsed single field plus coordinate vectors).  Pre-v4.15.0
   callers doing `E, x, y = create_gaussian_schell_source(...)` now
   silently bind `E.ndim == 3` and `x` to a scalar `dx`.  Pass
   `return_kind='ensemble'` explicitly to acknowledge the new
   contract, or `return_kind='mcf'` to opt into a
   `PartialCoherenceMCF` object instead.  v4.15.2 (P0-NEW-1 closure)
   emits a `DeprecationWarning` on the default path; removal in v5.0.
2. **`strehl_vector` default reference removed**: v4.15.0 had a
   buggy default plane-wave reference that produced unity Strehl for
   any uniform field of equal power AND `Strehl > 1` on focused PSFs
   (the focused field is more peaked than the plane-wave reference at
   matched total power).  v4.15.1 requires the caller to pass
   `reference=` explicitly; the docstring also drops the unverified
   "Richards-Wolf high-NA" claim.  See P1-F1-3 below.
3. **`system.evaluate` mixed-shape prescription raises `ValueError`**:
   a prescription containing BOTH `surfaces` + `thicknesses` AND
   `elements` + `all_thicknesses` keys previously silently picked one
   schema.  v4.15.1 rejects it at the validator with a clear message.
   Callers passing raw Zemax-loader output need to filter the
   surfaces keys before handing the dict in:
   ```python
   rx_filtered = {k: v for k, v in rx.items()
                  if k not in ('surfaces', 'thicknesses')}
   ```
   See P1-F1-6 below.

### P0 closures

**P0-NEW-1 — `make_off_axis_parabola` factory fix (doubly broken):**
Decenter formula corrected `f*tan(alpha)` -> `2*f*tan(alpha)` (chief-
ray geometry on parent paraboloid).  Tilt remains 3-tuple
`(off_axis_angle, 0.0, 0.0)`; factory docstring now loudly documents
that the OAP prescription is **intended for `apply_real_lens_traced`
exclusively** — the paraxial `apply_real_lens` cannot interpret the
3-tuple tilt correctly.  New end-to-end raytrace test
(`test_end_to_end_raytrace_focuses_at_offset`) pins the off-axis
focal-point location to within 1% of the chief-ray geometric
prediction at 60° fold (α=π/6).  v4.15.2 (P3 doc-drift fix): the
original CHANGELOG cited a pi/4 fold -- the actual test uses π/6
because α=π/4 is degenerate at this geometry.  P3 carryover:
`vertex_radius` now validated (must be `None` or finite positive).

**P0-NEW-2 — Schell-family factories redesign:**  The v4.15.0
factories collapsed the `n_realizations` ensemble into a single
fully-coherent complex field before return — the documented
partial-coherence contract was unfulfillable.  v4.15.1 introduces a
hybrid:

* Default `return_kind='ensemble'`:  factory returns
  `(ensemble, dx, dy, wavelength)` where `ensemble` has shape
  `(n_realizations, Ny, Nx)`.  Caller iterates over realizations and
  averages intensities downstream — physically-correct partial
  coherence.
* Opt-in `return_kind='mcf'`:  factory returns a new
  `PartialCoherenceMCF` dataclass with `.intensity()`,
  `.coherence_at(...)`, and `.coherent_modes()` methods.  For small
  grids (`Ny*Nx <= 64**2`), stores the full `J(r1, r2)`; for larger
  grids, stores the leading K coherent modes (Wolf 1982 JOSA
  decomposition) via SVD of the ensemble matrix.  Truncation
  threshold:  smallest K with `cumsum(eigvals)/sum(eigvals) >= 0.99`
  (Karhunen-Loève default).
* **Physics fix:**  the random-phase RMS normalization (which forced
  `sigma_phi = 1` regardless of `sigma_g`) is replaced with the
  spec-correct Fourier-filtered Gaussian-noise recipe.  Now `sigma_g
  -> 0` actually approaches incoherent (off-diagonal MCF -> 0) and
  `sigma_g -> infinity` approaches coherent (rank-1 MCF).

Affected factories: `create_gaussian_schell_source`,
`create_schell_model_source`, `create_annular_incoherent_source` +
matching `Source.gaussian_schell` / `Source.schell_model`
classmethods.  Note: MCF-aware downstream propagators are deferred
to v4.16+; the `PartialCoherenceMCF` object is consumable for
analysis / inspection in v4.15.1.

### P1 closures (Tier 0 audit recommendations)

* **P1-NEW-A: `sparrow_resolution` canonical Sparrow root-finding.**
  Implementation rewritten to true two-source dip-vanishing condition
  `d²/dr² [PSF(r-d/2) + PSF(r+d/2)]_{r=0} = 0` via
  `scipy.ndimage.map_coordinates` sub-pixel azimuthal averaging +
  cubic-spline 2nd derivative + `scipy.optimize.brentq` root-finder.
  Now returns 2.273 µm vs expected 2.273 µm at lambda=600nm, f/#=4
  (previously 1.93 µm, 15% low).
* **P1-NEW-C: 7 UI Source-factory deprecation callsites migrated** to
  kwarg-only canonical form in `lumenairy/ui/model.py`.  The v4.15.0
  release that introduced the deprecation shim now also migrates its
  own internal UI consumers.
* **P1-NEW-D: Raytrace flat-keys allowlist** at `raytrace/core.py:
  1507-1521` extended with `q_bfs_coeffs`, `q_con_coeffs`, `r_max`.
  Forbes Q prescriptions in flat-keys form no longer silently drop
  the coefficients at the gather step.
* **P1-NEW-E: Zemax `.zmx` QBFS/QCON parsing** added to
  `io/prescriptions.py`.  `.zmx` files with Q-type freeforms now
  load with `freeform_type='q_bfs'` or `'q_con'` + coefficients +
  `r_max` (parsed from `DIAM` / `PARM` lines), instead of silently
  degrading to base conic.
* **P1-F1-1: Q-bfs/Q-con radial-clip alignment.**  v4.15.0's
  rectangular `|X| <= norm_x AND |Y| <= norm_y` clip let pixels at
  `(0.9*r_max, 0.9*r_max)` (radial `r = 1.27*r_max`) through — outside
  the Forbes domain.  v4.15.1 uses a radial primary clip
  `r <= r_max` with the rectangular `(norm_x, norm_y)` box as
  secondary aperture.
* **P1-F1-2: `surface_sag_freeform` requires `r_max`** for
  `freeform_type in ('q_bfs', 'q_con')`.  Previously defaulted
  silently to `r_max=1.0` (a unit-mismatch bug — user passing X/Y
  in metres got sag computed on a sub-pixel of the actual aperture).
  Now raises `TypeError` with a clear message.
* **P1-F1-3: `strehl_vector` default-reference removed** (breaking).
  v4.15.0's default plane-wave reference produced unity for ANY
  uniform field of equal power AND Strehl > 1 on focused PSFs
  (more peaked than plane-wave at equal total power).  v4.15.1
  requires explicit `reference=` and softens the docstring
  (drops unverified "Richards-Wolf high-NA" claim).
* **P1-F1-4: `rayleigh_resolution` Gaussian-PSF false-positive fixed.**
  Now requires a strict subsequent rise above the candidate minimum
  by >=0.5% of peak before declaring first zero; Gaussian-like PSFs
  (no true zero) return NaN + `RuntimeWarning` advising
  `fwhm_resolution` / `sparrow_resolution` instead.
* **P1-F1-5: `astigmatism_mag_angle` docstring range correction** —
  `(-pi/4, pi/4]` -> `(-pi/2, pi/2]` (the actual range from
  `0.5 * atan2(c3, c5)`).
* **P1-F1-6: `system.evaluate` mixed-shape `ValueError`** — a
  prescription with both `surfaces`+`thicknesses` AND
  `elements`+`all_thicknesses` keys is now rejected at the
  validator with a clear message rather than silently picking a
  schema.  Behaviour change: callers passing raw Zemax-loader
  output need to filter the surfaces keys (`{k:v for k,v in rx.items()
  if k not in ('surfaces','thicknesses')}`).
* **P3-NEW-A: Forbes Q wave-optics path** — v4.15.0's
  `apply_real_lens` silently `RuntimeWarning`d and skipped the
  Forbes Q freeform contribution.  v4.15.1 routes Q-bfs / Q-con
  through `surface_sag_freeform` properly (option (a) of the audit
  recommendation), inheriting P1-F1-1 + P1-F1-2 guards.

### P2 closures

* **`__all__` symmetry**:  `surface_sag_q_bfs`,
  `surface_sag_q_con` re-exported from
  `lumenairy/elements/__init__.py`; `make_off_axis_parabola` from
  `lumenairy/io/__init__.py`.  `from lumenairy.elements import
  surface_sag_q_bfs` now works.
* **Sentinel consolidation**:  `_ZeroApertureMaskSentinel`
  (`optimize/core.py`) and `_AngleUnsetSentinel`
  (`elements/polarization.py`) now inherit from `_deprecation._Sentinel`
  base class.  `_Sentinel` gained pickle-safe `__reduce__` + name-keyed
  `_SENTINEL_REGISTRY` + `_sentinel_unpickle` reconstructor so
  pickle round-trips return the singleton instance (not a fresh
  sentinel).
* **`system.evaluate` Zemax-shape test** added (the audit's "headline
  ergonomic claim was untested" finding closed).

### P3 closures

* `n=1.0` consistency:  `optimize/multiconfig._resolve_lens_glass_index`
  bounds widened from exclusive `(1.0, 5.0)` to inclusive
  `[1.0, 4.0]` matching `register_fixed_glass`.
* Codegen runtime version pin gains an upper-bound major-version
  warning (`UserWarning` if running on `lumenairy >= 5.0.0`).
* `LambertianBSDF.evaluate` gains explicit surface-frame docstring
  + `RuntimeWarning` if `incident_direction` is non-axially-aligned
  without explicit frame transform.
* Coatings TIR cap warnings promoted from filtered `RuntimeWarning`
  to always-emit `UserWarning`.
* Forbes Q orthonormalizer docstring formula corrected:
  `c_n = sqrt((2n+3)(n+2)/(n+1)^2)` -> `c_n = sqrt((2n+3)(n+2)/(n+1))`
  (the implementation was already correct; only the docstring lied).
* `astigmatism_mag_angle` docstring range correction (also P1-F1-5).
* CHANGELOG/release-notes: lenses_maslov `_ZERO_APERTURE_MASK`
  sentinel branch now lives at `optimize/wrapper_merits.py:855`
  (the `if _cache['mask'] is _ZERO_APERTURE_MASK` line); was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split (the
  branch moved out of the monolithic core.py to the new
  ``wrapper_merits.py`` submodule that hosts the ``MultiWavelengthMerit``
  / ``MultiFieldMerit`` / ``ToleranceAwareMerit`` triplet); was
  `:3015` pre-v4.16.3 Agent C `Constraint` auto-probe
  DeprecationWarning latch + pickle catch widening (~17 lines added
  above the sentinel branch); was `:2974` pre-v4.16.2 Agent B
  `MultiWavelengthMerit` `FutureWarning` latch + Constraint-probe
  move + lambda pickle-probe (~41 lines added above the sentinel
  branch); was `:2958` pre-v4.16.1 Agent A `MultiWavelengthMerit`
  `SUM`->`AVG` refactor (~16 lines added in the merit-aggregation
  block above the sentinel branch); was `:2980` pre-v4.15.4 Agent
  B `_PerturbedABCDFallbackSentinel` deletion (~55 lines removed
  at the top of the sentinel block); and `:2905` pre-v4.15.3
  sentinel-wiring work.  The remaining
  sentinel class + singleton (`_ZeroApertureMaskSentinel` /
  `_ZERO_APERTURE_MASK`) are at `optimize/context.py:74` and
  `optimize/context.py:84` respectively post-v5.1.0 Agent E 6-file
  split (was `optimize/core.py:2044` and `optimize/core.py:2054`
  pre-split, post Agent E's v4.15.2 `_Sentinel` base-class
  refactor).  v4.15.2 (P3):
  citation refreshed after a
  second line-drift pass against the current source supersedes
  the earlier stale citations.

### CLUSTER_B Item 6 — `rays_from_field` bridge function

New `lumenairy.rays_from_field(E, *, dx, wavelength, dy=None,
n_rays=200, placement='cdf', angle_method='complex_gradient',
intensity_threshold=1e-4, z0=0.0, random_state=None) -> RayBundle`
samples a coherent field into a geometric ray bundle.  Bridges
`propagators/` (wave) <-> `raytrace/` (ray) so users can overlay
ray traces on coherent-field plots, seed a Maslov/GBD bundle from a
measured pupil field, or hand a coherent field into the geometric
ray tracer for hybrid analysis.

Placement modes: `'cdf'` (separable inverse-CDF, fast),
`'rejection'` (true 2-D rejection, exact), `'uniform'` (grid + threshold
mask).  Angle methods: `'complex_gradient'` (phase-ratio central
difference, singularity-safe — adapted from spec for correct
behaviour at Nyquist), `'unwrap_gradient'` (np.unwrap-based, fragile
near vortices).  Evanescent rays (`L² + M² > 1`) flagged with
`RAY_EVANESCENT` and `alive=False`.

13 tests + 1 runnable example
(`examples/rays_from_pupil_field.py`).  Implementation note: 3
spec deviations (phase-ratio central difference instead of literal
`Im(grad E / E)`; evanescent test samples 6x Nyquist to avoid
spectral aliasing; OPD test uses wrap-free phase slope) all
documented in the agent's release notes.

### CLUSTER_B Item 2 — Operator algebra

New `lumenairy/algebra/` subpackage implementing Nazarathy/Shamir
operator algebra (JOSA 70 (2), 1980).  9 new symbols at top level:

* `Operator`, `CompositeOperator` — base classes; ABCD-tracking
  algebraic composition.
* `FreeSpace(d, *, method='auto')`, `ThinLens(f)`,
  `CylindricalLens(f_x, f_y)`, `Magnify(a_x, a_y)`,
  `FourierTransform(f_focal)` — primitive operators.
* `Aperture(diameter, shape)`, `GaussianAperture(sigma)` — passive
  aperture operators (identity ABCD).
* `Operator.from_prescription(prescription, wavelength)` —
  prescription-dict -> CompositeOperator factory.  Paraxial-only;
  produces ABCD identical to `system_abcd(...)` to within 1e-12 abs.

Composition: `A * B` means "first B, then A".  ABCD of `A * B` is
`A.abcd @ B.abcd`.  Application: `sys(source) -> Source`, or
`sys.apply(E, dx=..., wavelength=...)`.  Anamorphic support via
separate `_abcd_x` / `_abcd_y`.

91 tests + 2 runnable examples (`examples/algebra_4f_system.py`,
`examples/algebra_anamorphic.py`).  Spec deviation: `Magnify._apply`
uses closed-form `sqrt(a_x*a_y)` amplitude prefactor instead of
spec's `resample_field` recipe (the spec recipe had an energy-
conservation bug; closed-form preserves energy per-pixel by
construction).  Phase 2 symbolic reduction (FreeSpace+ThinLens+
FreeSpace collapse, etc.) explicitly deferred to a future PR.

### Test counts

* Pre-v4.15.1 baseline (v4.15.0): 1425 unit tests + 1 skip + 1 xfail.
* v4.15.1 additions (`pytest --collect-only` items, parametrised
  test cases counted separately): A=19, B=20+migrated, C=13+migrated,
  D=18, E=20, F=20, G=91; gross 201 collected, net ~200 added (the
  +200 number nets out test migrations: agents B and C migrated
  v4.15.0 property pins to v4.15.1 ensemble / analytical-value pins).
  v4.15.2 (P2 + P3 test-count refresh): the F=20 count supersedes
  the original CHANGELOG's F=13 -- Agent F shipped 7 additional
  follow-up tests beyond the initial 13 enumerated in
  `.release_notes_v4_15_1_agent_f.md`, bringing the file to 20
  pytest items.  The A=19 count likewise supersedes the original
  A=18.
* Final: **1625 unit tests passing, 1 skipped, 1 xfailed**; **34/34
  validation files passing**.

### Deferred to v4.16+

* Modal-asymptotic independent ground-truth pin against
  `propagate_hf_chebyshev_quadrature(method='direct')` (audit
  P1-NEW-B, Tier 1).  v4.15.0 replaced known-buggy warm-start with
  unverified cold-start; this pin closes the verification gap.
* 4 remaining V2 meta-pin candidates (sentinel-aware branch
  propagation, `_xp_of` dispatch, `dy` parameter threading,
  `__all__` symmetry walker).
* MCF-aware downstream propagators (consume `PartialCoherenceMCF`
  through propagation chains).
* Multi-process atomic-append for `storage.py` (HDF5 SWMR + Zarr
  distributed lock).
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
  carryover).
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support.

---

## [4.15.0] — 2026-05-18

**Major minor release** rolling together carryover P1s from the
v4.14.2 audit (the "v4.14.4" patch scope) + ROADMAP v4.15 + ROADMAP
v4.16 into a single coordinated ship.  **1425 unit tests pass** (up
from 1265; +160 net new pins), **1 documented skip**, **1 documented
xfail** (`create_led_source` validation entry-point exemption);
**34/34 validation files pass**.

### Headline: modal-asymptotic 19.4x perf win + wrong-saddle physics fix

`propagate_modal_asymptotic` switches from a per-pixel warm-started
Newton loop to a single batched cold-start
`_solve_envelope_stationary_batch` + `_compute_M_b_batch` path
(the private helpers v4.14.0 shipped but did not consume on the
public path).  Closes the v4.14.0 audit's "wrong-saddle basin"
physics finding (warm-start chain entered wrong-saddle basins at
grid edges, silently zeroing those pixels via the `|b_quad| > 700`
overflow guard).  Cascade impact: any caller that builds on
`propagate_modal_asymptotic` (aberration tensor sigma-integration
grid path, through-focus / polychromatic helpers per focus or
wavelength point) gains the perf win and the grid-edge correctness
in one step.

Output is **bit-different** from v4.14.x at grid edges (strictly
more non-zero pixels because the cold-start finds the physical
saddle uniformly).  Four bit-equal pins migrated to property
pins (1e-8 abs vs cold-start reference + 5% energy + nz-count
>= warm-start ref):
`test_lg00_single_mode_matches_reference`, `test_multimode_matches_reference`
in `test_perf_v4_12_0_asymptotic.py`, plus
`test_lg00_single_mode_bit_equal` and `test_lg_p0_4mode_prescription_bit_equal`
in `test_audit_fixes_v4_14_0_agent_1.py`.  The v4.14.1 row-reset
warm-start pin (`test_row_reset_resets_warm_start`) was retargeted
to the v4.15 stronger structural guarantee:  the scalar
`solve_envelope_stationary` is no longer invoked by the public
path in any `maslov_tracking` mode (the warm-start chain is
structurally deleted, not just reset).

Measured: **19.4x** speedup at N=128 LG_(0,0); +52 non-zero
pixels recovered on the same grid (15918 vs 15866).

### Source factory normalisation + ergonomic system entry

Pre-v4.15 the 5 `Source.method` classmethod factories had
inconsistent positional order — some put size-arg first, others
N first.  v4.15 picks the canonical order
`Source.method(*, N, dx, wavelength, <size_kwargs>)` (kwarg-only).
The legacy positional form still works for one release with a
`DeprecationWarning` routed through the new
`_deprecation.warn_deprecated_signature` helper with
`version_removed='5.0'`.  Affected factories: `Source.gaussian`,
`Source.plane_wave`, `Source.point_source`, `Source.top_hat`,
`Source.fiber_mode`.

New ergonomic entry `lumenairy.system.evaluate(prescription,
source, *, output_grid=None, output_dx=None, ...)` (also
top-level `lumenairy.evaluate`).  Accepts both Zemax-loader
prescription shape (`elements` + `all_thicknesses` keys) and
factory shape (`surfaces` + `thicknesses` keys).  Users loading
a `.zmx` file no longer have to build the element list manually
before propagating.

### 7 new public-API functions (ROADMAP v4.16 closure)

* `ee_polychromatic(prescription, wavelengths, weights, radii, ...)`
  — convenience chain over `polychromatic_psf` +
  `encircled_energy_radius`.
* `strehl_vector(Ex, Ey, Ez=None, *, reference=None)` —
  vector Strehl with optional `Ez` z-component (Richards-Wolf
  high-NA case).
* `coupling_efficiency_vector(Ex, Ey, Ez=None, *, mode_Ex,
  mode_Ey, mode_Ez=None, dx)` — vector overlap integral with a
  vector mode.
* `rayleigh_resolution(psf, dx, wavelength, *, axis='radial')`
  — first-zero-of-PSF Rayleigh diffraction limit.
* `sparrow_resolution(psf, dx, *, axis='radial')` — empirical
  Sparrow criterion (dip-just-vanishes for two overlapping
  point sources).
* `fwhm_resolution(psf, dx, *, axis='radial')` — twice the
  FWHM half-radius of the central peak.
* `astigmatism_mag_angle(coeffs)` — Mahajan §8.2 conversion of
  Zernike `(c3, c5)` to `(|astig|, theta)` in the
  OSA/ANSI convention matching `zernike_decompose`.

All 7 are top-level exports in `lumenairy.__all__`.

### 3 new source factories (partial-coherence + ring incoherent)

* `create_gaussian_schell_source(*, N, dx, wavelength, w0,
  sigma_g, n_realizations=16, ...)` — spatially-incoherent
  Gaussian-Schell beam via random-phase ensemble.
* `create_schell_model_source(*, N, dx, wavelength,
  intensity_profile, coherence_length, n_realizations=16, ...)`
  — generic Schell-model (caller supplies intensity profile).
* `create_annular_incoherent_source(*, N, dx, wavelength,
  inner_radius, outer_radius, n_realizations=16, ...)` —
  angular-spectrum ensemble with finite source extent for
  partial-coherence integration (distinct from existing
  monochromatic-coherent `create_annular_beam`).

Matching `Source.gaussian_schell(...)` / `Source.schell_model(...)`
classmethods.  All 3 call `_validate_grid_params` in the first
10 lines (per the v4.14.2 audit's input-validation entry-point
meta-pin candidate).

### Forbes Q-type freeform basis + off-axis parabola factory

* `surface_sag_q_bfs(X, Y, *, radius, coefficients, r_max, ...)`
  — Forbes Q-bfs basis (Forbes 2007, *Opt. Express* 15(8) 5218,
  eq. 13).  Best-fit-sphere subtracted; orthonormal on the
  weight `u^2 (1-u^2) d(u^2)` over `[0, 1]`.
* `surface_sag_q_con(X, Y, *, radius, conic, coefficients,
  r_max, ...)` — Forbes Q-con basis (Forbes 2010,
  *Opt. Express* 18(13) 13851, eq. 6).  Conic-subtracted;
  orthonormal on weight `u^4 d(u^2)`.

Implementation uses the shifted-Jacobi 3-term recurrence
(A&S 22.7.1) on `t = 2x - 1` for `x = u^2 in [0, 1]`:
Q-bfs `(alpha, beta) = (1, 1)` with orthonormaliser
`c_n = sqrt((2n+3)(n+2)/(n+1)^2)`; Q-con `(alpha, beta) =
(0, 2)` with `c_n = sqrt(2n+3)`.  Orthonormality verified
numerically to <1e-6 over the first 5 orders for both bases.

* `make_off_axis_parabola(focal_length, off_axis_angle,
  clear_aperture, *, glass='__MIRROR__', vertex_radius=None,
  name=None) -> dict` — prescription factory for OAP segments.
  Single parabolic surface (conic `k = -1`, vertex radius
  `R = 2*focal_length`) with `decenter` and `tilt` set to the
  parent-axis offset and local-frame tilt.

### v4.14.2 carryover P1s closed (the "v4.14.4" scope)

**UI subpackage (P1-UI-1 through P1-UI-7):**
* `main_window.py` glass table now includes `N-LASF9` and
  `S-NPH1` (P1-UI-1) and `_nudge_distance` routes through
  `set_display_distance` (P1-UI-2, coordinate-mode aware).
* `model.py` undo state-capture now includes
  `wavelength_weights`, `field_weights`, `lens_options`
  (P1-UI-3).  Back-vertex calculation consolidated into a
  single `_prev_element_back_vertex_world` helper to prevent
  drift between the 717-718 vs 752 sites (P1-UI-4).
* `waveoptics_dock.py` re-parent now guarded by
  `shiboken6.isValid(original_parent)` to avoid the
  mid-dialog segfault risk (P1-UI-5).
* `psf_mtf_dock.py` ray-traced OPD now accumulates via
  `np.add.at(... mean)` instead of last-write-wins per pixel
  (P1-UI-6); out-of-aperture rays are filtered by a bounds
  mask before indexing instead of `np.clip(...).astype(int)`
  silently snapping to the pupil edge (P1-UI-7).

**Codegen + ghost + glass:**
* P1-CG: generated scripts now embed a runtime version pin
  (`if tuple(...) < (4, 15, 0): raise RuntimeError(...)`) plus
  a `lumenairy_version:` comment stamp.
* P1-GH-1: `non_sequential_stray_light` accepts a `seed: int |
  None = None` kwarg (default `None` uses system entropy so MC
  produces a real uncertainty band; pass a fixed integer to
  pin reproducibility).
* P1-GL-1: bundled Sellmeier rows for `SiO2` / `F_SILICA` /
  `FUSED_SILICA` (Malitson 1965 fused-silica coefficients) and
  `S-LAH64` / `S-LAH79` (OHARA Zemax 2017-11-30 catalog from
  the refractiveindex.info-database YAML).  Minimal installs
  without `refractiveindex` now resolve these glasses through
  the bundled Sellmeier fallback.

### Exhaustive P2/P3 sweep + meta-pin candidate #3

User-requested exhaustive enumeration of the v4.14.2 audit's
18 P2 + 12 P3 findings.  Net closure tally:

* 7-8 of 18 P2 closed (4 by Agent F directly: `lumenairy_context`
  6/7 redundant clears, `create_multi_field_sources` factory-
  validation list, ROADMAP refresh, HDF5/Zarr `lumenairy_version`
  attr stamping; 3 by other agents in this release:
  `_validate_grid_params` bool reject, deprecation-shim
  `_deprecation.py` migration with `version_removed`, codegen
  version pin; 1 partial: artifact version pinning — HDF5+Zarr
  done, codegen done).  10-11 P2 deferred (architectural items
  reserved for v4.16+ or v5.0).
* 4 of 12 P3 closed by Agent F: CHANGELOG line-citation drift
  (3 stale `optimize/core.py:...` ranges in the v4.14.2 entry
  refreshed to current line numbers — see commit diff for the
  exact pre/post values); `create_led_source` legacy-shim
  error-message clarity; README `makedammann2d _legacy_units='SI'`
  migration example; README cookbook section with examples for
  the 6 v4.14.0 public functions.

**New structural meta-pin (V2 candidate #3):**
`tests/unit/test_v4_15_dispatcher_pin_validate_grid_params.py`
walks every `create_*` factory and asserts `_validate_grid_params`
appears in the first 15 body lines.  17 PASS + 1 documented
xfail (`create_led_source` legacy-shim positions validator past
the head window — pinned via `xfail(strict=True)` so future
refactors that lift the validator forward flip to XPASS and
the exemption is removed).

### Version-stamping on HDF5 / Zarr writes

`io/storage.py` now writes a `lumenairy_version` attr on every
`create_dataset` / `create_array` / `create_group` site (7
locations).  Future-proof for cross-version field-file
compatibility checks.

### Bundled-glass registry reverse-direction consistency check

The v4.14.2 `_check_glass_registry_consistency` only walked
registry-entry -> `SELLMEIER_COEFFICIENTS` (forward).  v4.15
adds the reverse walk: every key in `SELLMEIER_COEFFICIENTS`
must appear in `GLASS_REGISTRY` (with `'__sellmeier__'` flag
if it's pure Sellmeier).  Coefficient rows added without a
registry entry would have remained silent dead code; this
catches them at module load.

### ROADMAP refresh

`ROADMAP.md` updated to v4.15.0 baseline.  v4.14.1 / v4.14.2 /
v4.14.3 / v4.15.0 entries added to Shipped highlights;
items closed in v4.15 removed from v4.15 + v4.16 target lists
and renumbered.  v4.16+ target list reseeded with the items
deferred from the v4.14.2 audit + the 5 V2 meta-pin
candidates.

### Test counts

* Pre-v4.15 baseline (v4.14.3): 1265 unit tests.
* v4.15 additions: A=8, B=40, C=26, D=23, E=27, F=36
  parametrized tests across two new files; net 160 added.
* Final: **1425 unit tests passing, 1 skipped, 1 xfailed**
  (`_ZARR_MKDIR_PATCH_LOCK` exemption + `create_led_source`
  validation-entry exemption); **34/34 validation files
  passing**.

### Deferred to v4.16+

* 4 V2 meta-pin candidates still standing (sentinel-aware
  branch propagation, `_xp_of` cross-backend dispatch,
  `dy` parameter threading, `__all__` symmetry).  Input-
  validation entry-point candidate #3 is shipped this
  release.
* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  + distributed Zarr lock).  Single-process atomicity is
  documented in v4.14.3.
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1
  P1-I carryover).
* Modal-asymptotic JAX-twin lift to use the v4.15 batched
  helpers.
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support beyond the rotationally-symmetric Q-bfs /
  Q-con bases shipped here.
* Architectural items reserved for v5.0 (file splits, CI
  gates, shim removal — see ROADMAP).

---

## [4.14.3] — 2026-05-17

**Closes the v4.14.2 audit (`docs/audits/AUDIT_V4_14_2_2026_05_17.md`).**
The audit found 2 NEW P0 findings (`storage.py` non-atomic
`n_planes` increment → silent data loss in concurrent / Zarr
streaming; `makedammann2d` >1mm SI heuristic silently mangles
legitimate mm-scale gratings), 21 NEW P1s, 18 P2s and 12 P3s.
The "fix N, miss N+1" sibling-gap meta-finding recurred 5 ways
on v4.14.2.  v4.14.3 closes both P0s, the 5 sibling-gap
recurrences, 11 latent-bug P1s (1 real physics error in
`multiconfig.py`), and all 3 doc-drift P1s.  **1265 unit tests
pass** (up from 1190); **34/34 validation files pass**.

### Breaking changes — none

Two near-breaking deltas with explicit opt-in / opt-out:

* **`makedammann2d` accepts `_legacy_units='auto'|'um'|'SI'`** (default
  `'auto'` preserves the v4.14.2 per-parameter deprecation heuristic).
  The `'auto'` and `'SI'` modes now raise `ValueError` on any
  unit-bearing kwarg > 1.0 m (rejects nm-scale-garbage from the
  silent-mangling regime).  Legitimate >1m / mm-scale SI gratings
  set `_legacy_units='SI'` to bypass the legacy heuristic
  entirely.  Pure-legacy callers can set `_legacy_units='um'` to
  rescale all unit-bearing inputs by 1e-6 without firing the
  deprecation warning.
* **`create_led_source` legacy-positional shim hardened** with a
  scale-inversion sanity check: a canonical-order positional
  call (`N, dx, wavelength, diameter, divergence`) that
  accidentally slots a wavelength into the `_legacy_diameter`
  position now raises `TypeError` with a migration message
  instead of producing 633 nm "diameter" / 0.1 m "wavelength"
  garbage.

### P0 closures

**P0-NEW-1: `storage.py` n_planes atomicity** — `append_plane_h5`
(`io/storage.py:444`) and `_zarr_append_plane` (`:782`) bumped
`grp.attrs['n_planes']` AFTER `create_dataset`.  A crash between
the two operations left an orphan dataset; the next append used
the stale `n` to compute `plane_{N:02d}` and on Zarr (`overwrite=
True` at line 769) the orphan was silently clobbered.  Concurrent
appenders racing on `n_planes=N` both wrote `plane_{N:02d}`, the
second silently winning.  v4.14.3 inverts the ordering — attr
written BEFORE dataset create, try/except rollback on failure —
and drops `overwrite=True` on the Zarr path so the orphan case
now raises rather than silently destroying data.  Single-process
atomicity is documented; multi-process locking (HDF5 SWMR /
distributed Zarr lock) deferred to v4.15+.  3 regression tests
pin attr-write ordering, docstring contract, and the no-silent-
clobber invariant.

**P0-NEW-2: `makedammann2d` >1m upper-bound** — v4.14.2's
`value > 1e-3` heuristic silently rescaled mm-scale SI gratings
(coarse industrial Dammann, THz/MMW) by 1e-6 → nm-scale garbage.
v4.14.3 adds a `_legacy_units` kwarg (see "Breaking changes"
above) plus an explicit `ValueError` for any unit-bearing input
> 1.0 m in `'auto'`/`'SI'` mode.  3 regression tests cover the
upper bound, explicit `'SI'` mm-scale pass-through, and explicit
`'um'` rescale without `DeprecationWarning`.  3 historical test
sites that relied on the silent rescale were migrated to
`_legacy_units='um'`; one validation case (`test_elements.py`'s
100 µm legacy-µm Dammann) likewise opts in explicitly.

### P1 closures — sibling-gap recurrences (5)

* **P1-NEW-1: `clear_asm_caches()` LG-polynomial chain.**  v4.14.2
  chained 5 sibling caches but missed `_lg_polynomial_items`
  (`asymptotic.py:284`).  v4.14.3 adds the lazy-import + call
  pattern to `clear_asm_caches`, expanding its docstring to list
  all 8 caches it now drains.  Combined-drain test extended to
  assert `_lg_polynomial_items.cache_info().currsize == 0` after
  `clear_asm_caches()`.
* **P1-NEW-2: `apply_rotator` conflict-resolution symmetrised
  across 5 polarization helpers.**  v4.14.2 added `angle_deg=`
  conflict detection to `apply_rotator` only, leaving 4 sibling
  helpers (`apply_polarizer`, `apply_waveplate`,
  `apply_half_wave_plate`, `apply_quarter_wave_plate`) silently
  letting `angle_deg` overwrite `angle`.  v4.14.3 introduces a
  module-level `_AngleUnsetSentinel` singleton (matching the
  v4.14.1 `_ZeroApertureMaskSentinel` pattern) and a single
  `_resolve_angle(func_name, angle, angle_deg)` helper used by
  all 5 helpers.  The "explicit `angle=0` + `angle_deg=90`"
  conflict (which the v4.14.2 `angle != 0.0` check missed) now
  raises `ValueError` from a single canonical site.  19
  regression tests including a 5-way parametrized conflict
  matrix.
* **P1-NEW-4: `create_led_source` `*args` footgun closed** via
  scale-inversion sanity check in the legacy-positional shim.
  Rejected PEP 570 positional-only marker because it would have
  broken every existing kwarg-based call site (including the
  v4.14.2 audit-test infrastructure).  The new check fires on
  `apparent_wavelength > 10 * apparent_diameter` — catches the
  canonical-order mistake (`0.3 > 1.31e-6 * 10`) without
  triggering on legitimate legacy `(diameter, divergence,
  wavelength)` calls (real LED diameters > 10x their wavelength).
* **P1-NEW-5: `_validate_grid_params` tuple-N gating.**  v4.14.2's
  helper accepted both `int` and `(Ny, Nx)` tuples, but 7 of 10
  factories used `np.arange(N)` downstream and crashed with an
  opaque `np.arange` TypeError.  v4.14.3 adds a `support_tuple_N:
  bool = False` parameter; the 3 factories with genuine 2-D
  grid support (`gaussian_beam`, `hermite_gauss`,
  `laguerre_gauss`) opt in, the other 7 reject tuple-N with a
  clear `TypeError` at validation time.  10 parametrized tests
  across all 10 factories.

### P1 closures — under-examined modules (6)

* **P1-MC (real physics error): `multiconfig.py` hardcoded `n=1.5`**
  in the thin-lens / lensmaker formulae at `:265` and `:327`.
  Beam-expander and Keplerian-telescope multi-config builders
  computed lens powers assuming BK7-ish refractive index for
  every glass, silently producing wrong focal lengths for
  flint, SF, S-LAH, fused-silica, or any non-`n≈1.5` glass.
  v4.14.3 introduces `_resolve_lens_glass_index(name)` that
  routes through the canonical `glass.get_glass_index()` lookup
  with a documented `UserWarning` + bounded fallback (`n=1.5`)
  for genuinely unknown labels and bounds-check on custom
  callable returns (`1.0 < n < 5.0`).  Catches `(ValueError,
  KeyError, ImportError, RuntimeError, TypeError)` only —
  preserves the v4.13.0 Phase-2 narrowed-exception discipline.
* **P1-NEW-9: `create_bessel_beam` `cone_angle` constraint.**
  Now enforces `0 < cone_angle < pi/2` at construction time;
  `cone_angle = pi` previously produced `sin(pi) ≈ 0` and a DC
  field silently labelled "Bessel".
* **P1-NEW-10: `create_fiber_mode` `mode_field_diameter > 0`.**
  Now rejected at construction; negative MFD previously yielded
  a sign-flipped Gaussian, MFD=0 yielded an all-ones field.
* **P1-NEW-11: `surface_sag_xy_polynomial` / `surface_sag_chebyshev`
  negative-`norm_x`/`norm_y` rejection.**  `norm_x = -0.05`
  (typo on a 50 mm half-aperture) made `outside = abs(X) >
  -0.05` true everywhere, silently zeroing the entire freeform
  contribution.  Both branches now reject negative norms at
  function entry with a clear `ValueError`.
* **P1-GH-2: `ghost.py` IndexError on `elements`-key
  prescriptions.**  `ghost_analysis` and
  `non_sequential_stray_light` previously crashed on the two
  `elements`-style prescription schemas (surface-style entries
  with `'element_type'` and propagate-style entries with
  `'type'`).  v4.14.3 introduces a `_ghost_surfaces(prescription,
  wavelength)` adapter that detects each schema and routes
  through the appropriate canonical surface-expansion path.
  5 regression tests across all three schemas + missing-keys
  error path.
* **P1-GL-2: `register_fixed_glass` input validation.**  Name
  must be a non-empty string; `n` must be finite and in
  `[1.0, 4.0]`; overwriting an existing entry emits a
  `UserWarning`.  Si (n=3.4) and Ge (n=4.0) verified accepted;
  invalid forms rejected.

### P1 closures — doc-drift (3, retroactive to v4.14.2)

Per the v4.14.2 audit Part 2 / P1-NEW-6/7/8:

* **Agent D test count** (CHANGELOG line 374): `25 tests` →
  `8 tests + 1 parametrize entry`.
* **"6 factories" → "5 factories"** in 3 CHANGELOG sites
  describing the v4.13.x source-factory dispatcher pin.
* **`lenses_maslov.py` line-drift correction note** refreshed
  (`678/737/884/993` → `619/728/874`; verified against the
  current module via def-header grep).  Recorded that one of
  the 4 prior sites was consolidated away.
* **Audit-path references** (11 expected → 11 fixed): all
  `` `AUDIT_V4_*.md` `` in CHANGELOG (8 sites) and README (3
  sites) now carry the `docs/audits/` prefix added by the
  v4.14.2 doc reorganization.
* **Cache-locks meta-pin count** corrected `38 tests` → `39
  collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK`
  skip)` in CHANGELOG lines 158 / 186 and README line 110.

### Test counts

* Pre-v4.14.3 baseline (v4.14.2): 1190 unit tests.
* v4.14.3 additions:
  - Agent A (storage + makedammann + LG chain): 9 tests.
  - Agent B (sources + multiconfig): 25 tests.
  - Agent C (polarization + freeform + ghost + user_library):
    41 tests.
  - Agent D (doc corrections): 0 (paper-only).
* Final: **1265 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation
  files passing**.

### Deferred to v4.15+

* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  client + distributed Zarr lock).  v4.14.3 documents single-
  process guarantees and the multi-process restriction; the
  full multi-writer story is a v4.15+ ergonomics item.
* 18 P2 + 12 P3 findings from the v4.14.2 audit (UI module
  audit, codegen ergonomics, secondary doc-drift, additional
  factory validation gaps).  These remain catalogued in
  `docs/audits/AUDIT_V4_14_2_2026_05_17.md` for v4.15+ triage.

---

## [4.14.2] — 2026-05-17

**Closes the v4.14.1 audit (`docs/audits/AUDIT_V4_14_1_2026_05_17.md`).**  The
audit found 1 NEW P0 (a v4.11.2 regression carryover: `glass.py`
S-LAH64/S-LAH79 dispatch broken for 3 releases) plus 10 new P1s
(4 of which are "fix N, miss N+1" recurrences on v4.14.1 itself —
the aperture=0 sentinel missed 2 of 4 call sites, 7 older caches
still unlocked, 4 residual `0+0j` literal sites, `clear_asm_caches`
chain narrower than docstring) plus 6 P1s in under-examined modules
(freeform domain guard, makedammann unit drift, polarization API
inconsistencies, source factory validation gaps).  v4.14.2 closes
the P0 and all 10 P1s, adds **two new structural meta-pins** to
extend the v4.14.1 cache-clear dispatcher-pin pattern (cache↔lock
pairing + `0+0j` literal sweep), and closes a follow-up cache-lock
gap discovered by Agent C's meta-pin.  **1190 unit tests pass** (up
from 911); 34/34 validation files pass.

### Breaking changes — none

Two near-breaking output deltas with deprecation warnings:

* **`makedammann2d` is now SI metres throughout** (per-parameter
  deprecation heuristic detects legacy µm input and warns).  Pure-
  legacy calls and hybrid `periodx=20.0 (legacy µm) +
  waveln=1.31e-6 (already SI)` calls both continue to work; a
  `DeprecationWarning` directs users to migrate.
* **`create_led_source` signature reordered** to v4.7-canonical
  `(N, dx, wavelength, *, diameter, divergence_angle, dy=None, ...)`
  with a legacy-positional shim that emits a `DeprecationWarning`.

Neither breaks existing scripts but both warn on the legacy form.

### P0 closure — `glass.py` S-LAH64 / S-LAH79 dispatch

**3-release carryover regression.**  v4.11.2 (round-3 audit, CRIT-3)
removed S-LAH64 / S-LAH79 from `SELLMEIER_COEFFICIENTS` after
discovering the in-code coefficients were off by ±5.8% vs the Ohara
catalog, intending to route them through `refractiveindex.info` —
but `GLASS_REGISTRY` was never updated.  Both glasses remained
flagged `'__sellmeier__'`, and the dispatcher at `glass.py:410-415`
raised `ValueError` on every call.  `ui/main_window.py:1552`
references S-LAH64 as a "known-good preset" — UI broken for 3
releases.

**Fix:** Routed both to `('specs', 'OHARA-optical', '<name>')` tuple
form (correct catalogue book name verified by introspection;
`'OHARA'` was a wrong first attempt).  Numerical verification:
n_d=1.7880 matches Ohara catalog n_d=1.78800 to 5e-5 for S-LAH64;
S-LAH79 returns n_d=2.0033 matching the catalog 2.00330.

**Structural counter-measure:** Module-load consistency check at the
end of `glass.py` walks `GLASS_REGISTRY` and raises `RuntimeError`
if any `'__sellmeier__'` entry is missing from
`SELLMEIER_COEFFICIENTS`.  Future drift fails fast at import time.

### P1 closures — sibling-gap recurrences on v4.14.1

* **P1-NEW-1: aperture=0 sentinel finish** (was incomplete in
  v4.14.1).  The v4.14.1 sentinel fix updated 3 callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) but missed 2:
  `ToleranceAwareMerit.evaluate` (`optimize/core.py:2805-2810`,
  refreshed v4.15.1 -- earlier ranges drifted by ~15 lines after
  Agent E's `_Sentinel` base-class refactor) and
  `MatchIdealSystem._make_source` (`optimize/core.py:977-991`,
  refreshed v4.15.0).
  Both used `_cache['E_ones'].copy()` without the `mask is
  _ZERO_APERTURE_MASK` branch, producing a full-grid plane wave
  instead of zero on `aperture_diameter=0`.  v4.14.2 adds the
  sentinel-aware branch (option (b) of the audit recommendation —
  explicit-per-call-site, matches the canonical 2 already-fixed
  sites).  **Investigation finding:** `apply_perturbations` only
  mutates per-surface `decenter` / `tilt` / `form_error`, NEVER
  the prescription-level `aperture_diameter`, so the audit's worry
  about perturbed-to-zero apertures is not triggered by existing
  code — pinned the contract via
  `test_apply_perturbations_does_not_modify_aperture`.
* **P1-NEW-2: 7 older caches now locked** following the v4.14.1
  pattern.  Locks added to `_ZERNIKE_BASIS_CACHE`,
  `_THROUGH_FOCUS_SCAN_JAX_CACHE`, `_PROPAGATE_SYSTEM_JAX_CACHE`,
  `_GS_KERNEL_CACHE`, `_ER_KERNEL_CACHE`, `_HIO_KERNEL_CACHE`,
  `_TRACE_JAX_CACHE`.  Lock-scope discipline matches v4.14.1: lock
  held for `get` / `move_to_end` ops, released before expensive XLA
  jit-compile / numpy basis build, re-acquired for insert + evict.
  Concurrent 4-thread tests confirm no exceptions, no `RuntimeError:
  dictionary changed size during iteration`, final state consistent.
* **P1-NEW-3: `clear_asm_caches()` chain expanded** to 5 sibling
  caches via lazy-import + call (`clear_zernike_basis_cache`,
  `clear_through_focus_scan_jax_cache`,
  `clear_propagate_system_jax_cache`, `clear_phase_retrieval_caches`,
  `clear_trace_jax_cache`).  Docstring rewritten to honestly list
  all caches the function now clears.  Pinning test populates each
  cache then calls `clear_asm_caches()` and asserts emptiness.
* **P1-NEW-4: 2 P1-severity residual `0+0j` sites** swept
  (`optimize/merit_terms.py:515` post v5.1.0 Agent E 6-file split;
  was `optimize/core.py:987` pre-split via Agent B's P1-NEW-1 work
  + `analysis/phase_retrieval.py:402`; the optimize citation was
  refreshed `966 -> 987` in v4.15.0 to match the post-v4.14.2
  drift, then `core.py:987 -> merit_terms.py:515` in v5.1.0).  Now use the `np.zeros((), dtype=...)` pattern.  **Structural pin (new meta-pin):**
  `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` walks
  every `lumenairy/*.py` file (117 modules) and asserts no
  unallowlisted `np.where(..., 0+0j)` literal — three exemption
  layers (pure-comment lines, trailing `.astype()` recovery
  pattern, explicit P3 allowlist for `ui/psf_mtf_dock.py:230`).

### P1 closures — under-examined modules

* **P1-NEW-5: `surface_sag_xy_polynomial` domain guard** added.
  v4.14.1's freeform XY polynomial branch evaluated `c * X**i *
  Y**j` for every pixel — a high-order term like `(2,0): 1e3` on a
  50-mm half-grid produced 2.5 m of sag at the corner, propagating
  into raytraced rays outside the physical aperture.  v4.14.2 adds
  `norm_x, norm_y` kwargs + an `xp.where(<inside_box>, sag, 0.0)`
  clip matching the Chebyshev branch.  Backward-compatible (default
  `norm_x = norm_y = 1.0` plus raw-coordinate polynomial evaluation
  — no coefficient rescaling).
* **P1-NEW-6: `makedammann2d` SI conversion** with per-parameter
  deprecation heuristic.  See "Breaking changes" above.
* **P1-NEW-7: `apply_rotator` accepts `angle_deg=`** kwarg-only,
  matching the v4.7 polarization-family convention.  Conflict-
  resolution policy: `angle_deg` and non-zero `angle` (radians)
  with disagreement → `ValueError`.
* **P1-NEW-8: `JonesField.__init__` input validation**.  Added
  positive-finite `dx, dy` checks + 2-D shape check.  A 1-D field
  accidentally passed in now raises a clear `ValueError` at
  construction time, not an opaque FFT failure downstream.
* **P1-NEW-9: `create_led_source` signature drift** closed.  See
  "Breaking changes" above.  Underlying `create_top_hat_beam`
  handles `dy != dx` natively, so no `ValueError`-on-anisotropy
  raise was needed (unlike the JAX-twin precedent).
* **P1-NEW-10: `_validate_grid_params` helper** added at the top of
  `sources/core.py`.  Applied to all 10 factories
  (`create_gaussian_beam`, `create_hermite_gauss`,
  `create_laguerre_gauss`, `create_top_hat_beam`,
  `create_annular_beam`, `create_fiber_mode`, `create_led_source`,
  `create_bessel_beam`, `create_point_source`,
  `create_tilted_plane_wave`).  60 parametrized tests confirm
  `ValueError` on N≤0, dx≤0, wavelength≤0, non-finite inputs.

### Follow-up sibling-gap closed in same release

The v4.14.2 cache↔lock meta-pin (Agent C's new structural pin)
discovered a previously-unflagged single-cell lazy-init cache in
`propagators/asymptotic.py:_JAX_IFT_SOLVER_CACHE`.  Race window
on first concurrent call decorates the JAX `custom_vjp` solver
twice.  v4.14.2 closes the gap in the same release: added
`_JAX_IFT_SOLVER_CACHE_LOCK` with double-check locking pattern
(fast path no lock for the common populated-cache case; slow path
acquires lock, re-checks, delegates the actual build to
`_build_jax_ift_solver_impl`).  Meta-pin's
`_KNOWN_CACHE_SIBLING_GAPS` set is now empty — future regressions
land there.

### Two new structural meta-pins

Extending v4.14.1's cache-clear dispatcher-pin pattern to two more
sibling-gap classes the audit identified:

* **`test_v4_14_2_dispatcher_pin_cache_locks.py`** — 39 collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK` skip):
  walks every library module via `pkgutil.walk_packages`, finds
  names matching `^_.*_CACHE$`, asserts each has a corresponding
  lock (accepts both `_FOO_LOCK` and `_FOO_CACHE_LOCK` naming
  conventions).  Reverse check: every `_LOCK` has a corresponding
  cache (or matches the documented `_PATCH_LOCK` exemption for
  `_ZARR_MKDIR_PATCH_LOCK`).
* **`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`** (123 tests)
  — walks all `lumenairy/*.py` files for `np.where(.*, 0+0j)`
  literals.  Exemption layers for pure-comment lines, trailing
  `.astype()` dtype-recovery pattern, and explicit P3 allowlist for
  `ui/psf_mtf_dock.py:230` (audit-rated low-priority UI site).

### Retroactive doc-drift corrections to v4.14.1 entry

Per audit Part 1.3:

* Agent D test count (v4.14.1 line ~372): 25 → 8 (bottom-line 911 was correct).
* Source-factory dispatcher count: previously claimed "all 6" → corrected to 5 factories (3 CHANGELOG sites updated).
* `lenses_maslov.py` line-drift correction note refreshed.

### Test counts

* Pre-v4.14.2 baseline (v4.14.1): 911 unit tests.
* v4.14.2 additions:
  - Agent A (glass + freeform + polarization): 13 tests
  - Agent B (aperture=0 sentinel finish): 10 tests
  - Agent C (7-cache locks + clear_asm scope + meta-pin): 57
    tests pass (19 fix-pins + 39 meta-pin parametrizations
    collected, of which 38 pass + 1 documented
    _ZARR_MKDIR_PATCH_LOCK skip)
  - Agent D (sources + DOE + meta-pin): 199 tests (76 fix-pins +
    123 meta-pin parametrizations)
* Final: **1190 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation files
  passing**.

### Deferred to v4.15+

Row_reset physics pin against `propagate_hf_chebyshev_quadrature
(method='direct')` (audit Tier-1 #12 — contract-pinned in v4.14.1,
not numerically-pinned).  Backend dispatch (`_xp_of`) on the 6 new
v4.14.0 analysis functions.  Modal asymptotic per-pixel
vectorisation public switch.  Source factory signature
normalisation (the LED factory landed in v4.14.2; the rest still
have mixed positional/keyword conventions).  `system.evaluate
(prescription, source, ...)` ergonomic entry.  See
[`ROADMAP.md`](ROADMAP.md) for the full forward plan.

## [4.14.1] — 2026-05-17

**Closes the v4.14.0 audit (`docs/audits/AUDIT_V4_14_0_2026_05_17.md`).**  The
audit found 1 P0 (silent-wrong physics in the 77× LG/HG mode-stack
cache key) + 6 P1s + 10 P2s + 8 P3s + 7 doc-drift items.  v4.14.1
closes the P0, **all 6 P1s** (including P1-NEW-5 `row_reset`
Newton warm-start via the coordinated option-(a) fix across the
public path and 3 reference-loop pins), the top-priority P2s (cache
locks, monkey-patch removal, final `0+0j`/`1+0j` sweep,
`fiber_mode` in the dispatcher pin), and all 7 doc-drift items (4
retroactively corrected in the v4.14.0 entry below; 3 closed by the
v4.14.1 fixes themselves).  **911 unit tests pass** (up from 858);
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage.  Aperture-zero semantics restored to the
pre-v4.14.0 behaviour (which had silently flipped) — see P1-NEW-1
below.

### P0 closure — `_LG_MODE_STACK_CACHE` / `_HG_MODE_STACK_CACHE` cache key

The 77× perf win in v4.14.0 had a silent-wrong-physics bug.  The
cache key omitted `dx, dy` — only `(p_max, ell_max, Ny, Nx, w, cx,
cy, dtype_str)`.  Two calls with the same shape but different
physical pitch (e.g. `dx=1e-6` then `dx=2e-6`, both at N=256)
collided on this key: **the second call returned the FIRST grid's
modes evaluated against the second call's field.**  Silently wrong
overlaps on:

* wavelength-adaptive grid sweeps (re-pitch the grid per
  wavelength to keep `λ/dx` constant)
* multi-resolution analysis (debug at coarse `dx`, then fine `dx`)
* optimisation loops where `dx` is a free variable

**Fix:** `dx, dy` added to both LG and HG mode-stack cache keys.
Regression pin asserts that two calls at the same `N, w, cx, cy,
p_max, ell_max` but different `dx` return distinct mode stacks.

### P1 closures

* **P1-NEW-1: aperture=0 semantics regression.**  v4.14.0's
  `_get_wrapper_merit_cache` mapped `aperture <= 0` to `mask=None`,
  which downstream callers interpreted as "no clipping, full grid
  plane wave."  **Semantics flipped 180°** from the pre-v4.14
  behaviour where `aperture=0` produced an all-False mask (block
  all light).  v4.14.1 adds a `_ZeroApertureMaskSentinel` singleton;
  `_get_wrapper_merit_cache` returns the sentinel for `ap <= 0` and
  `None` for `ap is None`; callers compare via `is` and route the
  sentinel through an explicit all-zeros path.  Three callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) updated.
* **P1-NEW-2: Brewster-angle phase aggregation bug** in
  `coatings.py` carried into v4.14.0's wavelength batch.  The
  v4.13.1 audit flagged that `0.5 * (np.angle(r_s) + np.angle(r_p))`
  is off by π/2 or π at Brewster (~56°) because `r_p` sign-flips
  through zero.  v4.14.0's wavelength-batch rewrite inherited the
  bug verbatim.  v4.14.1 changes the aggregation to
  `np.angle(0.5 * (r_s + r_p))` (complex sum then angle — robust
  to π-discontinuities).  Reference helper in
  `test_audit_fixes_v4_14_0_agent_2.py` also updated to match
  (the audit explicitly anticipated this collateral update).
* **P1-NEW-3: `_solve_envelope_stationary_batch` contract
  violation.**  The function's docstring promised
  `converged_mask=False` for failed pixels (singular Hessian or
  non-finite update) but the code set `True` to drop them from the
  active set.  Currently dead production code; the contract is
  preserved for future callers.  v4.14.1 introduces a separate
  `finished` mask for active-set tracking, leaving `converged`
  to mean what the docstring says.
* **P1-NEW-4: `clear_lg_mode_stack_cache` now in top-level `__all__`.**
  v4.14.0's CHANGELOG claimed this was "Public" but the import
  wasn't in `lumenairy/__init__.py` — the audit-meta-finding
  ("fix N sites, miss N+1") recurring on the very release that
  shipped 80 dispatcher pins.  v4.14.1 closes the loop AND adds a
  new structural counter-measure: a parametrized **cache-clear
  dispatcher pin** (`tests/unit/test_v4_14_1_dispatcher_pin_cache
  _clears.py`) that walks every submodule's `__all__` for
  `clear_*` names and asserts each is re-exported at top level
  AND callable from `la.*`.  Future cache-clear additions can
  no longer regress this gap.
* **P1-NEW-6: `encircled_energy_radius` docstring** corrected.
  v4.14.0 claimed `ee[0] = 0 always` (in an inline comment, not
  the docstring).  Reality: `ee[0]` equals the cumulative power
  at radius 0 (the centre-pixel intensity contribution).  Docstring
  expanded to document the hot-centre behaviour explicitly;
  pinning test exercises a delta-like centre-pixel input and
  confirms the threshold-zero short-circuit returns 0.

* **P1-NEW-5: `row_reset` resets the Newton warm-start.**
  v4.14.0's `row_reset` branch reset Maslov-branch state
  (`last_arg_detM`, `maslov_branch`) at each raster row wrap but
  left the Newton warm-start `last_v_star` chaining across the
  discontinuous jump from (x_max, y_n) to (x_min, y_{n+1}) —
  plausibly the mechanism behind the v4.14.0 "wrong-saddle-basin"
  finding near grid edges (largest jump in s_2).  v4.14.1 chooses
  **option (a)** of the audit recommendation: the `row_reset`
  branch now resets `last_v_star = (v_cx, v_cy)` at each row wrap
  too.  Coordinated with the fix, the bit-equal pin in
  `test_audit_fixes_v4_14_0_agent_1.py::test_lg00_single_mode
  _bit_equal` and the older 1e-10 rel pins in
  `test_perf_v4_12_0_asymptotic.py::TestPropagateModalAsymptotic
  Correctness` were updated — their inline scalar references
  also reset `last_v_star` at row wrap so the bit-equality holds
  against the new physics-correct behaviour.  The v4.14.1 marker
  test (formerly `TestRowResetDoesNotResetWarmStart`, now
  `TestRowResetResetsWarmStart`) flipped its assertion to pin the
  new contract.

### Tier-2 closures

* **Thread-safety locks** on the three new v4.14.0 caches.
  `_LG_MODE_STACK_CACHE`, `_HG_MODE_STACK_CACHE`, and
  `_WRAPPER_MERIT_CACHE` now have module-level `threading.Lock`
  guards on their read-modify-write ops, mirroring the
  `_ASM_CACHE_LOCK` precedent in `propagation.py`.  Concurrent
  `OrderedDict.move_to_end` / `popitem(last=False)` from parallel
  `design_optimize` threads no longer race.
* **Monkey-patch removed** in `optimize/core.py`.  The v4.14.0
  pattern monkey-patched `propagation.clear_asm_caches` to chain
  in `_clear_wrapper_merit_cache`; v4.14.1 replaces this with a
  lazy-import + call inside `propagation.clear_asm_caches()`
  itself.  Eliminates re-import recursion risk and the case
  where importing `propagation` without `optimize` leaves the
  wrapper-merit cache resident.
* **LG/HG mode-stack cache wired into `clear_asm_caches()`** (not
  just `lumenairy_context()` as v4.14.0 had).  Same lazy-import
  pattern.  CHANGELOG claim from v4.14.0 now matches code.
* **Final `0+0j`/`1+0j` literal sweep**: 2 sites caught by the
  audit — `lenses_maslov.py:448` (in `sample_E_bilinear`) and
  `_lens_thin.py:173` (aplanatic branch, actually `1.0+0.0j` not
  `0.0+0.0j` — replaced with `xp.ones((), dtype=...)`).
* **`fiber_mode` added** to
  `TestP1CSourceFactoryDispatcherPin` parametrize list (v4.13.2's
  `create_fiber_mode` dy-widening was complete but the test
  list was stale).

### Retroactive CHANGELOG corrections to v4.14.0 entry

4 confirmed doc-drift items from the audit corrected in the
v4.14.0 entry below:

* "16 tests at 1e-10 rel" → 3 tests in the cited class
  (`TestPropagateModalAsymptoticCorrectness`), 16 across the
  file overall.
* "6 batched helpers consumed by 77× win" → helpers ship privately
  but currently have zero production consumers; reserved for the
  v4.15+ coordinated public switch.
* HG cache key documented as `w[s]` shorthand → both LG and HG
  keys spelled out (HG has both `wx` AND `wy`; v4.14.1 adds
  `dx, dy`).
* `lenses_maslov.py` line cites — drifted 4-10 lines since
  v4.14.0 ship; in-line correction note added.

The 3 other doc-drift items (LG cache wired into `clear_asm
_caches`, public `clear_lg_mode_stack_cache`, "dispatcher pin
covers all 5 factories") **became true** in v4.14.1 — no
retroactive edit needed; the v4.14.0 entry's claims are now
accurate.

### Test counts

* Pre-v4.14.1 baseline (v4.14.0): 858 unit tests.
* v4.14.1 additions:
  - Agent A (asymptotic): 8 tests
  - Agent B (optimize + propagation): 11 tests
  - Agent C (coatings + lens sweep): 8 tests
  - Agent D (exports + meta-pin): 8 tests + 1 parametrize entry
    + 17 cache-clear meta-pins
* Final: **911 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

Modal asymptotic per-pixel vectorisation public switch, Source
factory signature normalisation, `system.evaluate(prescription,
source, ...)` ergonomic entry.  See [`ROADMAP.md`](ROADMAP.md)
for the full forward plan.

## [4.14.0] — 2026-05-17

**Phase B of the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`).**
v4.13.2 closed Tier-0 (the 12 P1s + 5 cross-survey P0s + thin-lens
sibling sweep).  v4.14.0 closes Tier-1: the 7 Tier-1 perf wins from
audit Part 3, the top user-facing API gaps from the cross-survey,
and the 3 parametrized dispatcher pin families that close out the
sibling-gap audit-meta-finding.  All 858 unit tests pass; 34/34
validation files pass.

### Breaking changes — none

No user-facing breakage in v4.14.0.  All additions are net-new
public functions or internal perf optimisations behind unchanged
signatures.

### Performance wins (Tier-1 from audit Part 3)

| Hot path | Workload | Old | New | Speedup |
|---|---|---|---|---|
| `coating_reflectance` wavelength batch | 50 layers × 200 wv | 78.6 ms | 3.19 ms | **24.6×** |
| `decompose_lg` / `decompose_hg` cache (warm) | 256², p_max=3, ell_max=3 | 270 ms | 3.5 ms | **77×** |
| `MultiWavelengthMerit` meshgrid cache | N=512, 5 wl, 20 FD | (ref) | (ref/6.17) | **6.17×** |
| `_evaluate_polynomial_4d_and_grad34` | M=70, 16×16 grid | 646 µs | 171 µs | **4.6×** (typical configs) |
| `MultiWavelengthMerit` meshgrid cache (small) | N=128, 3 wl, 5 FD | (ref) | (ref/4.16) | **4.16×** |
| `ToleranceAwareMerit` meshgrid cache | N=128 | (ref) | (ref/3.17) | **3.17×** |
| Shack-Hartmann gather loop | K=4096, sa=8 | 7.2 ms | 3.2 ms | **2.27×** |
| Phase-retrieval `np.angle`/`np.exp` round-trip | GS, N=256, 50 iters | 1230 ms | 543 ms | **2.26×** |
| Phase-retrieval (ER variant) | same workload | 380 ms | 205 ms | **1.85×** |
| Phase-retrieval (HIO variant) | same workload | 426 ms | 268 ms | **1.59×** |
| `MultiFieldMerit` (limited by `np.exp`/`np.where`) | N=128 | (ref) | (ref/1.20) | **1.19×** |

**Meshgrid build count** in `Multi*Merit` / `ToleranceAwareMerit`:
from `O(n_wl × n_field × FD_evals)` (~1025 at N=512, 5 wl, 5 field,
20 FD) to **1 per `(N, dx, aperture)` signature** for the entire
optimisation run.  Cache cleared by `clear_asm_caches()`.

**Coating Snell-chain hoist** — the documented `n.imag-dropped-at-
Snell-step` approximation makes the per-layer chain wavelength-
independent.  v4.14 walks it ONCE outside the polarisation loop
instead of `n_wv × n_pol` times.  This is what unlocks the 24×
batched speedup.

**LG/HG mode-stack cache** (`_LG_MODE_STACK_CACHE` +
`_HG_MODE_STACK_CACHE`, both `OrderedDict` + LRU(32)).  LG key is
`(p_max, ell_max, Ny, Nx, w, cx, cy, dtype_str)`; HG key has both
`wx` AND `wy` (9-element tuple).  **Correction (v4.14.1):** v4.14.0
omitted `dx, dy` from these keys, producing silently-wrong overlaps
on multi-resolution or wavelength-adaptive grid sweeps; v4.14.1
adds `dx, dy` to both keys and wires the caches into
`clear_asm_caches()` (v4.14.0 only wired them into
`lumenairy_context(clear_caches_on_exit=True)`).  Public
`clear_lg_mode_stack_cache()` for explicit flushes (became
top-level-importable as `lumenairy.clear_lg_mode_stack_cache` in
v4.14.1).

**Phase-retrieval algebraic identity.**
`exp(1j * np.angle(F)) == F / np.abs(F)` for nonzero `F`.  v4.14
replaces the two-transcendental round-trip in the NumPy paths of
GS / ER / HIO with the divide, eliminating ~4 trig ops per pixel per
iteration.  JAX paths already had the optimisation; only NumPy
paths were touched.

### Modal asymptotic per-pixel vectorisation — NOT shipped publicly

Audit opportunity #1 (target 20-100×) was investigated and turned
out to be **subtler than the audit estimate**.  The brief asked to
vectorise `propagate_modal_asymptotic`'s per-pixel Newton solve.
The vectorised cold-start batched Newton finds the physical saddle
uniformly across all output pixels; the pre-v4.14 warm-started
Newton chains the previous pixel's `v_star` and at grid-edge pixels
lands in a **wrong-saddle basin** that produces `|b_quad| > 700`,
which the overflow guard zeros.

The pre-v4.14 behaviour is pinned bit-equal by `tests/unit/test_perf
_v4_12_0_asymptotic.py::TestPropagateModalAsymptoticCorrectness` (3
tests in the cited class, 16 in the file overall, all at `1e-10
rel`).  The vectorised cold-start produces strictly
more non-zero pixels (physically more correct) but breaks the
existing pin.  **This is a real physics finding** worth a coordinated
v4.15+ release that updates the test pin alongside the algorithm
change.

Shipped in v4.14.0: 6 new private vectorised helpers
(`_solve_envelope_stationary_batch`, `_compute_M_b_batch`,
`_phi_v2_hessian_batch`, `_gaussian_moment_table_2d_batch`,
`_batched_polynomial_substitute_linear_2d`, `_batched_polynomial
_under_affine_shift`).  The public `propagate_modal_asymptotic` body
is unchanged.  **Correction (v4.14.1):** the helpers ship privately
but currently have zero production consumers; the 77× LG/HG mode-
stack cache reaches into `lg_polynomial`/`hg_polynomial`/`_evaluate
_poly2d` directly, not through the batched helpers.  The helpers
are reserved for the v4.15+ coordinated public switch.

### New public API (cross-library survey gaps)

Six new functions exposed at `lumenairy.*` and in the appropriate
`__all__` tier:

* **`encircled_energy_curve(E, dx, *, dy=None, radii=None,
  centroid=None, n_radii=64) -> (radii, ee)`** — fraction of total
  power within radius `r` of the centroid.  Standard spec-sheet
  metric.  (Tier 4: Analyse.)
* **`encircled_energy_radius(E, dx, *, dy=None, threshold=0.84,
  centroid=None) -> float`** — radius enclosing the threshold
  fraction.  Default 0.84 matches the "84% encircled radius" lens
  spec convention.
* **`mtf_cutoff(mtf_profile, freq, *, threshold=0.5) -> float`** —
  spatial frequency at which a 1D MTF drops below threshold.
  Returns `np.inf` if the MTF stays above threshold for all
  frequencies.
* **`beam_diameter(E, dx, *, dy=None, threshold='1/e^2',
  centroid=None) -> float`** — diameter at which intensity drops
  below threshold.  String thresholds: `'1/e^2'`, `'1/e'`, `'FWHM'`,
  `'D4sigma'` (forwards to `beam_d4sigma` and returns geometric
  mean).
* **`depth_of_focus(wavelength, f_number, *, formula='rayleigh')
  -> float`** — one-sided depth of focus.  `'rayleigh'`:
  `±4 f#² λ`.  `'marechal'`: `±λ / NA²`.  Note: both formulas
  reduce to `4 f#² λ` (since `NA = 1/(2 f#)`); kept as separate
  named entries for derivation-clarity, cross-validated by test.
* **`plot_wavefront(opd, dx, *, dy=None, aperture=None,
  units='waves', wavelength=None, cmap='RdBu_r', show_stats=True,
  ax=None, fig=None, title=None) -> (fig, ax)`** — Zemax-style
  wavefront map: NaN outside the aperture, divergent colormap
  centred at zero, PV/RMS overlay annotation.  (Tier 9: Plotting.)

All six honour `dy` from the start (defaulting `dy=None → dy=dx`,
area integrations use `dx*dy`).

### Sibling-gap parametrized dispatcher pins (audit meta-finding closure)

The audit's recurring meta-finding ("fix swept N sites, missed
N+1") is closed for three family classes via parametrized pins.
**80 new dispatcher pins across 3 test files**.  Each pin enumerates
every reachable variant and asserts the same property; a future fix
that misses one variant fails CI at test-time.

* **`(scalar, vectorial) HFPI`** — 29 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_hfpi.py`).  Properties: `_spawn_rng`
  independence; grazing-ray `inf`/`NaN` guard; alive-mask
  correctness in `accumulate_*_to_grid`; public-surface
  importability across 13 names.
* **`(NumPy, JAX) apply_real_lens` family** — 35 pins (`tests/unit
  /test_v4_14_0_dispatcher_pin_apply_lens.py`).  Properties:
  `glass_after='MIRROR'` case-insensitive guard (15 parametrisations
  covering upper/lower/mixed-case); `dy=None` acceptance; `dy != dx`
  honour-vs-raise contract; complex64 dtype preservation across all
  5 variants.
* **Welford-mirror convention** — 16 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_welford_mirror.py`).  Properties: hand-computed
  analytical match for `seidel_coefficients` and `petzval_radius`;
  no-NaN for `aberration_summary` and `chromatic_focal_shift`;
  no-raise for `distortion_grid`, `field_aberration_sweep`,
  `eval_image_plane_wfe`; algebraic equivalence to the Welford
  formula.

### Sibling-gaps discovered by the dispatcher pins (now fixed)

The complex64 dtype-preservation pin tripped on 3 of 5
`apply_real_lens` variants — a previously-undetected sibling-gap
from the v4.13.2 B.4/B.5 thin-lens sweep.  Fixed in this release:

* **`apply_real_lens_maslov`** — `lenses_maslov.py`:
  - `_integrate_quadrature` (line 619) and `_integrate_stationary
    _phase` (line 728) and `_integrate_local_quadrature` (line 874)
    all hardcoded `dtype=np.complex128` allocations.  Each now
    accepts an `out_dtype` kwarg defaulting to `np.complex128` for
    back-compat; `apply_real_lens_maslov` threads `E_in.dtype`.
    (CHANGELOG line-cite refreshed in v4.14.2: prior CHANGELOG
    revisions cited drifted post-landing line numbers from the
    v4.14.0 tag and a further-drifted set from v4.14.1.  Current
    v4.14.2 def-header sites are 619/728/874 — one prior site was
    consolidated away during subsequent edits, leaving 3.)
  - The post-quadrature re-fit at line 566 multiplied by `1j`
    (complex128) which promoted the result; now cast back to
    `E_in.dtype`.
  - Final `normalize_output` step multiplied by a python float
    (float64) which silently promoted complex64 → complex128; final
    cast back to `E_in.dtype` added.
* **`apply_real_lens_traced_jax`** — `_lens_jax.py:573`: was calling
  `cdtype = _resolve_jax_complex_dtype()` which returns the library-
  default complex dtype (complex128 when `jax_enable_x64=True`),
  silently upcasting complex64 inputs.  v4.14 passes `E_in.dtype` to
  the resolver so the user's input dtype is honoured.
* **`apply_real_lens_maslov_jax`** — `_lens_jax.py:819`: same fix.

### Cache invalidation hygiene

`clear_lg_mode_stack_cache()` (new in v4.14 from the decompose_lg
cache work) wired into `lumenairy_context(clear_caches_on_exit=
True)` to match the existing pattern for the other 7 cache-clear
helpers.

### Test counts

* Pre-v4.14.0 baseline (v4.13.2): 710 unit tests.
* v4.14.0 additions:
  - Agent 1 (asymptotic): 15 tests in `test_audit_fixes_v4_14_0_agent_1.py`
  - Agent 2 (coatings + polynomial): 10 tests in `..._agent_2.py`
  - Agent 3 (phase-retrieval + SH): 13 tests in `..._agent_3.py`
  - Agent 4 (Multi*Merit cache): 18 tests in `..._agent_4.py`
  - Agent 5 (API gaps): 12 tests in `..._agent_5.py`
  - Agent 6 (dispatcher pins): 80 tests across 3 files (`test_v4_14
    _0_dispatcher_pin_hfpi.py`, `..._apply_lens.py`, `..._welford
    _mirror.py`)
* Final: **858 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

* **Modal asymptotic per-pixel vectorisation** (audit opportunity
  #1).  The wrong-saddle-basin physics finding noted above needs a
  coordinated test-pin update alongside the algorithm change.
  Vectorised helpers shipped privately for future use.
* **Source factory signature normalisation** (audit Stream B #10) —
  `Source.gaussian(w0, N, ...)` puts beam-size first; `Source.plane
  _wave(N, ...)` puts N first.  Picking a canonical order requires
  a deprecation window; rolling into v4.15 to coordinate with other
  Source-related cleanups.
* **`system.evaluate(prescription, source, ...)` ergonomic entry**
  — closes a one-liner UX gap (`propagate_through_system` currently
  takes an element list, not a prescription dict).

### Deferred to v5.0

Tier-2/3/4 structural items unchanged from v4.13.2 entry: six file
splits; CI gates; back-compat shim removal; shared Chebyshev
helpers extraction; audit-fix test-file consolidation by topic;
constrained optimisation; checkpoint/resume; CDGM/Hikari/Sumita
glass catalogues; off-axis conics in surface frame; Q-type
freeform.

## [4.13.2] — 2026-05-17

**Closes the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`) plus its
Part 10 consolidation with a parallel 6-agent cross-library survey.**
The v4.13.1 audit identified 12 new P1s (5 sibling-gap recurrences +
5 fresh-eyes bugs + 2 partial-closure follow-ups) plus 17 perf
opportunities and 22 structural recommendations.  The parallel
cross-library survey turned up an additional 5 P0s (in `optimize/`
and `io/`) and 7 P1s (mostly `dy` convention drift across analysis +
system + lens family).  v4.13.2 closes the full consolidated Tier-0
set.  All 710 unit tests pass; 34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.2.

### P0 closures (cross-library survey)

* **`make_lg_aberration_merit_jax` no longer silently ignores its
  `targets` dict.**  The function's inner loop body was literally
  `pass`; `wgt` was captured but never multiplied; the aberration-
  tensor call was outside the loop.  Public API exported from
  `lumenairy.__init__` — the function was returning the same
  `(0,0)`-piston sum regardless of input weights.  v4.13.2: now
  weights the `(0,0)` target correctly; non-`(0,0)` targets raise
  `NotImplementedError` with a clear migration message (the
  underlying `aberration_tensor_lg00_jax` only computes the piston
  coefficient).
* **`MultiFieldMerit.field_angles` accepts both scalars and `(theta_x,
  theta_y)` tuples.**  v4.13.1 applied Y-axis tilt only despite the
  docstring implying generic off-axis angle; `MatchIdealSystemMerit`
  took `(theta_x, theta_y)`.  v4.13.2 widens `field_angles` to accept
  both forms with a one-shot `DeprecationWarning` on the scalar form;
  internal storage normalises to tuples so the evaluate loop is
  uniform.  Non-zero `theta_x` now actually produces an X-axis tilt.
* **`load_plane_slice` documented return type matches actual.**
  Documented to return slice array; actually returns `(arr, attrs)`
  tuple in both HDF5 and Zarr paths.  Docstring + type annotation
  corrected to `Tuple[ndarray, Dict[str, Any]]`.
* **CODE V `.seq` round-trip preserves BFL.**  Reader was dropping the
  last refracting surface's `THI` (a legitimate CODE V BFL
  convention).  New top-level `'back_focal_length'` prescription key
  (float, SI meters) carries the BFL through round-trip; populated by
  both readers and exporters.
* **Quadoa `.qos` round-trip preserves BFL** (same fix as CODE V).

### Sibling-gap P1 closures (the audit's headline pattern recurring)

Five new "fix-swept-N-sites-missed-N+1" instances closed.  All five
have parametrized dispatcher-level pin tests preventing future
recurrence within the same family.

* **`vectorial_hfpi` RNG-correlation sibling.**  v4.11.2 fixed the
  scalar `hfpi.propagate_hfpi_freespace_aperture` via `_spawn_rng`;
  the vectorial sibling at `vectorial_hfpi.py:399` still passed the
  same `rng` to source-init AND aperture re-emission, perfectly
  correlating the diffraction events.  v4.13.2 ports `_spawn_rng`.
* **`subaperture` `decompose_lg(E_in, ...)` sibling.**  v4.11.2 fixed
  `hf.propagate_huygens_fresnel_through_prescription` to decompose
  the input field per-patch; `subaperture.propagate_subaperture
  _asymptotic` still passed `source_amplitudes={(0,0): 1.0+0.0j}`
  — every patch got an identical plane-wave-equivalent unit
  fundamental, ignoring `E_in` beyond a waist estimate.  v4.13.2
  ports the per-patch LG decomposition with three new kwargs
  (`source_lg_p_max=3`, `source_lg_ell_max=3`,
  `source_lg_amp_threshold=1e-6`) and amplitude-threshold pruning.
* **`petzval_radius` Welford-mirror convention sibling.**
  `seidel_coefficients` was fixed in v4.11.2 with the Welford `n2 =
  -n1` convention for mirrors; `analysis/field.py:petzval_radius`
  still skipped every mirror (because `n1 == n2` after the loader
  set `glass_after == glass_before`), silently dropping every
  mirror's Petzval contribution.  **Wrong by 100% for catadioptric /
  Cassegrain designs.**  v4.13.2 applies the Welford-parity
  convention; Cassegrain regression test pins against the analytic
  Mahajan formula.
* **`_build_jax_prescription` `glass_after='MIRROR'` sibling.**
  v4.13.1 P1-A added BOTH `is_mirror=True` AND case-insensitive
  `glass_after='MIRROR'` guards to `apply_real_lens`; the JAX
  prescription builder at `raytrace/jax_trace.py:649-669` only got
  the first.  Hand-built prescriptions with Welford-style mirror-
  via-glass-string slip through and are silently traced as
  refractive air→air.  v4.13.2 adds the second guard.
* **JAX lens twins thread `dy`.**  The two NumPy `apply_real_lens`
  variants accept `dy=None`; the JAX twins (`apply_real_lens_traced
  _jax`, `apply_real_lens_maslov_jax`) did not.  Anamorphic round-
  trip through `Source.dy → propagate_through_system_jax →
  apply_real_lens_traced_jax` silently dropped y-pitch at the JAX
  boundary.  v4.13.2: both JAX twins accept `dy=None` and raise
  `ValueError` on `dy != dx` (matching the existing NumPy
  precedent on the square-grid contract).

### Fresh-eyes P1 closures (audit Part 2.2)

* **`apply_mirror` NaN propagation.**  For a hyperbolic mirror with
  conic such that `(1+k)*h²/R² >= 0.9999`, sag → NaN → phase → NaN →
  `E *= NaN` poisons every pixel of the subsequent ASM step.
  `apply_real_lens:704-705` had the equivalent NaN-zeroing guard;
  `apply_mirror` did not.  v4.13.2 mirrors the guard.
* **`_zero_C_air_gap` raises on degenerate ABCD.**  When `abs(C1 -
  C0) < 1e-30`, the function silently returned the placeholder
  thickness from the input prescription (a non-afocal beam expander
  with zero combined power).  Callers (`beam_expander_prescription`,
  `keplerian_telescope`) catch `RuntimeError`; v4.13.2 raises it
  explicitly so the fallback fires.
* **`propagate_to_plane` inf/NaN positions guarded.**  For `Nz ≈ 0`
  and `z_target != z_curr`, the divisor went to `1e-30` →
  `t ≈ 1e30` → positions += inf.  The alive-mask correctly tagged
  these dead, but `paths.positions` still contained inf/NaN —
  downstream code reading positions without masking by `alive` was
  poisoned (e.g. `_hfpi_segment_trace`).  v4.13.2 zeroes the step on
  dead/grazing rays in BOTH `hfpi.propagate_to_plane` and
  `vectorial_hfpi.propagate_vector_to_plane`.
* **`RandomState.choice` int dtype aligned across backends.**  JAX
  path returned int32 (default for `jax.random.randint`); NumPy
  path returned int64.  v4.13.2 dispatches `jax.random.randint(...,
  dtype=jnp.int64)` so both backends agree.
* **`trace_prescription` uses `_surface_copy_with` instead of
  mutating shared `Surface.thickness`.**  When `image_distance=` was
  supplied, the function mutated the input prescription's last
  surface in place.  The `Surface` dataclass is shared with
  `surfaces_from_prescription`; downstream calls reused corrupted
  thickness.  v4.13.2 clones via `_surface_copy_with` (matching the
  pattern at `lens_abcd`).

### Partial-closure follow-ups from v4.13.1

* **`dual_annealing` callback wired into cancellation protocol.**
  v4.13.1 P2 #13 wired `CancellableProgress` into 4 scipy callbacks;
  the `dual_annealing` site used an unnamed inline lambda that did
  NOT poll `is_cancelled(progress)`.  Cancellation latency was
  unbounded for that one method.  v4.13.2 replaces the lambda with
  a named callback matching the pattern in the other three.
* **`RandomState.choice` old-JAX safety net.**  v4.13.1 P1-F closed
  the `replace=False` regression on JAX 0.10.0+ but the CHANGELOG
  promised a graceful old-JAX safety net the code lacked.  v4.13.2
  wraps the dispatch in `try/except TypeError` with a `RuntimeError`
  raising a clear "JAX >= 0.4.0 required" message.

### Cross-library survey P1 closures

* **`strehl_ratio` + `polychromatic_psf` accept `dy=`.**  The
  v4.13.0 L3 sweep added `dy` to `Source` / `PropagationResult` /
  free-space propagators but missed both Strehl helpers (using
  `dx**2` not `dx*dy`).  v4.13.2 adds `dy: Optional[float] = None`
  to both; back-compat preserved bit-for-bit when `dy is None`
  (the historic `dx**2` form is retained so existing tests with
  the FP-identity assumption stay green).
* **Wrapper-merit context threads `x=ctx.x`.**  `MultiWavelengthMerit`,
  `MultiFieldMerit`, `ToleranceAwareMerit` built sub-`Evaluation
  Context`s without `x=ctx.x`.  A `JaxMeritTerm(build_args=...)`
  wrapped inside any of these silently fell back to legacy
  `fn(ctx)` mode, degrading the analytic-gradient path to FD.
  v4.13.2 threads `x=ctx.x` across all three.
* **`propagate_through_system` element handlers thread `dy`.**
  Every element handler (lens, aperture, mask, zernike, mirror,
  etc.) passed `dx=dx` only — anamorphic `dy` was silently squared
  to `dx` on every non-`propagate_*` element.  v4.13.2 routes
  `current_dx` AND `current_dy` through all 13 element handlers.
* **`Source.fiber_mode` accepts `dy=` end-to-end.**  v4.13.1 P1-C
  threaded `dy` through 4 of 5 Source factories; `fiber_mode`
  remained a dead-`dy=` code path because `create_fiber_mode`
  didn't accept `dy=`.  v4.13.2 widens `create_fiber_mode` to
  accept `dy=` (forwarding to `create_gaussian_beam`); the
  dispatcher pin in `TestP1CSourceFactoryDispatcherPin` now covers
  all 5 factories.

### Thin-lens family sibling-gap sweep (cross-library survey)

The cross-library survey turned up additional sibling-gaps in the
thin-lens family — same v4.13.1 P3 #21 dtype-aware-zero fix that
landed in `apply_mirror` + `apply_aperture`, but missed across
multiple sites:

* **9 sites of `0.0+0.0j` complex128 literal** in `_lens_thin.py` (4
  sites) + `_lens_real.py` (5 sites).  Each was a `xp.where(..., E,
  0+0j)` clear-aperture or stop-mask construct that silently
  upcast complex64 → complex128.  v4.13.2 replaces each with
  `xp.zeros((), dtype=E.dtype)`.
* **6 thin-lens functions** (`apply_thin_lens`, `apply_spherical
  _lens`, `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply
  _grin_lens`, `apply_axicon`) constructed `xp.exp(1j * phase)`
  from float64 phase without dtype matching against the input
  field.  Same complex64→complex128 upcast as the `0+0j` literal,
  at a different call site.  v4.13.2 adds the dtype-coercion line
  to all 6 functions, matching the v4.13.0 L6 pattern in
  `apply_mirror`.
* **3 thin-lens functions** (`apply_cylindrical_lens`, `apply_grin
  _lens`, `apply_axicon`) lacked the documented `use_gpu` parameter
  entirely.  v4.13.2 adds the parameter and the canonical CuPy-
  dispatch pattern to all three.
* **Latent CuPy dispatch bug fixed in the other 3 thin-lens
  functions.**  `apply_thin_lens`, `apply_spherical_lens`,
  `apply_aspheric_lens` had the `use_gpu` parameter from v3.5.5
  but the CuPy dispatch was broken because the function bodies
  referenced bare `cp` — Python's LEGB name resolution doesn't
  consult module-level PEP 562 `__getattr__` for function-local
  lookups.  v4.13.2 routes all three through `_lenses_module.cp`
  (matching the working pattern in `apply_cylindrical_lens`).

### Quick wins

* **5 mis-tiered names in `__init__.py.__all__`** moved to correct
  tiers: `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`
  → Tier 1 (Build a system); `monte_carlo_tolerancing_jax`,
  `monte_carlo_tolerancing_linearized`, `tolerancing_report` →
  Tier 4 (Analyse).
* **Duplicate `reset_fft_backend` import** removed from `__init__.py`
  (was imported at both line 46 and line 59).
* **Wiki Release-Notes.md broken anchor** fixed:
  `[4.13.1](#whats-new-in-4-14-0)` → `[4.13.1](#whats-new-in-4-13-1)`
  (stale from the v4.14→v4.13.1 rename script during v4.13.1 ship).

### Test counts

* Pre-v4.13.2 baseline: 654 unit tests.
* v4.13.2 audit-response additions: 56 new tests across 4 new files
  (Agent A: 12 in `test_audit_fixes_v4_13_2_agent_a.py`; Agent B:
  20 in `..._agent_b.py`; Agent C: 14 in `..._agent_c.py`; Agent
  D: 10 in `..._agent_d.py`).
* Final: **710 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.14.0

Tier-1 perf opportunities from the v4.13.1 audit Part 3 (modal
asymptotic per-pixel loop 20-100×; multi-merit meshgrid cache
5-10×; phase-retrieval `angle`/`exp` round-trip 2-4×; coating
reflectance wavelength batch 5-15×; decompose_lg/hg per-mode
rebuild 3-8×; Shack-Hartmann gather loop 5-15×; `_evaluate
_polynomial_4d_and_grad34` 3-5×) plus the cross-library survey's
top user-facing API gaps (encircled energy, MTF cutoff frequency,
depth of focus, plot_wavefront, beam diameter at threshold) plus
parametrized dispatcher pins for the three audit-recommended
sibling families (`(scalar, vectorial) HFPI`, `(NumPy, JAX)
apply_*`, Welford-mirror convention).

### Deferred to v5.0

Tier-2/3/4 structural items: six file splits (`raytrace/core.py`
4422 LOC, `propagation.py` 3710, `asymptotic.py` 3597, `optimize
/core.py` 3258, `io/prescriptions.py` 2829, `analysis/core.py`
2196); CI gates (`ruff`, `mypy --strict`, unit-test PR job);
back-compat shim removal; shared Chebyshev helpers extraction;
audit-fix test-file consolidation by topic; constrained
optimisation; checkpoint/resume; CDGM/Hikari/Sumita glass
catalogues; off-axis conics in surface frame; Q-type freeform.

## [4.13.1] — 2026-05-17

**Closes the v4.13.0 audit (`docs/audits/AUDIT_V4_13_0_2026_05_17.md`) plus an
additional perf-survey pass.**  v4.13.0 was tagged in git but never
published to PyPI; that audit identified 7 P1 (latent bug), 9 P2
(code smell), and 6 P3 (cleanup) findings — most importantly 3
"sibling-gap" recurrences (the `apply_real_lens` mirror guard
missed its parent function; the L2 JAX dtype routing missed
`error_reduction_jax`/`hybrid_input_output_jax`; the L3 `dy`
threading missed `Source.propagate` + 5 classmethod factories).

v4.13.1 closes every Tier-0 and Tier-1 item from that audit, plus
all Tier-2 P2 follow-ups and the Tier-3 cleanups, plus 3 new perf
wins discovered in the parallel survey.  All 654 unit tests pass;
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.1.  The v4.13.0 breaking changes
(`rcwa.py` → `thin_grating.py` rename without back-compat shim,
and the `wavelength` sentinel raising `ValueError` on omission)
are inherited but do not regress further.  See v4.13.0 entry's
"Breaking changes" subsection below.

### Sibling-gap P1 closures (audit's headline finding)

**P1-A — `apply_real_lens` mirror guard.**  The v4.13.0 L4a sweep
hardened 4 sibling `apply_real_lens_*` variants (`_traced`,
`_traced_jax`, `_maslov`, `_maslov_jax`) but missed the parent
`apply_real_lens` itself.  A hand-built prescription containing
`surfaces[i]['is_mirror']=True` (and no `'elements'` key) would
silently misompute through `apply_real_lens` while raising
`ValueError` from the 4 siblings.  Ported the same pre-flight
guard from the `_lens_traced.py` template.  Added a parametrized
dispatcher-level pin (`TestL4aMirrorGuardDispatcherPin`) over all
5 variants to prevent this class of sibling-gap from recurring.

**P1-B — ER/HIO complex-dtype routing.**  `gerchberg_saxton_jax`
correctly routed `dtype` through `_resolve_jax_complex_dtype` in
v4.13.0, but `error_reduction_jax` and `hybrid_input_output_jax`
skipped the resolver — the EXACT silent float64 → complex64
demotion bug L2 was supposed to close, still live on those two
kernels.  Both now go through `_resolve_jax_complex_dtype` and
`_resolve_jax_real_dtype`; cache keys pinned on the resolved
complex dtype to match the GS pattern.  Parametrized dispatcher
pin (`TestP1BPhaseRetrievalDtypeResolver`) covers all 3 kernels.

**P1-C — `Source.propagate()` and 5 factories thread `dy`.**
`Source.propagate(...)` was constructing the result Source
without `dy=self.dy`, silently losing the y-pitch on every
anamorphic call.  Same gap in `Source.gaussian`, `plane_wave`,
`point_source`, `top_hat`, and `fiber_mode` — each factory
forwarded `dy` to its `create_*` helper (so the underlying
E-field WAS built on the anamorphic grid) but the final
`cls(E=..., dx=..., wavelength=..., ...)` call dropped `dy`.
Fixed in all 6 sites.  `Source.propagate` gained an optional
`output_dy` kwarg for symmetry with the existing `output_dx`
override.  Parametrized dispatcher pin
(`TestP1CSourceFactoryDispatcherPin`) covers the 4 dy-aware
factories; `fiber_mode` excluded because `create_fiber_mode`
itself does not accept `dy=` (pre-existing limitation outside
P1-C scope).

### UI + infrastructure P1 closures

**P1-D — `ThinGratingDock._run` kwargs mismatch.**  The dock was
calling `grating_efficiency_vs_wavelength(groove_index=...,
substrate_index=..., profile=..., angle=...)` but the function
expects `(period, depth, *, n_ridge, n_groove, n_substrate,
n_superstrate, order, ...)`.  Every dock click raised
`TypeError`, swallowed silently into the summary text box — the
dock was non-functional.  Rewrote `_run` with the correct
signature, added missing UI inputs for `n_ridge` and
`n_superstrate`, and extracted the math path into a pure
`_compute_efficiency_data(inputs: dict) → dict` helper so the
compute path is unit-testable without a live `QApplication`.  As
a side effect, the dock now does **one** `thin_grating_efficiency_1d`
call per wavelength (full per-order matrix) instead of N sweeps
through the single-order helper.

**P1-E — `_context.py` cache-clear import moved inside try.**
The `from .propagators.propagation import clear_asm_caches` was
OUTSIDE the try-block guarding the call.  If the import raised
(rename, circular import, partial install), the `ImportError`
bypassed not just that cache-clear but all 6 subsequent guarded
cache-clear blocks.  Moved inside the try, added `ImportError`
to the except tuple, matching the pattern used by the other 6
clears.

**P1-F — `RandomState.choice` on JAX honours `replace=False`.**
With `p=None`, the JAX path silently ignored `replace=False` and
returned with-replacement samples.  Now dispatches to
`jax.random.choice(sub_key, a, shape=shape, replace=False, p=p)`
for both weighted and unweighted branches (JAX 0.10.0 supports
the combination — verified at runtime).

**L7 — `test_bench_through_focus_scan_jax_first_vs_warm`** now
clears `_THROUGH_FOCUS_SCAN_JAX_CACHE` before timing the first
call, matching the 4 sibling benchmarks.  On a re-run within the
same process the reported "first call" timing is now the cold
path, not a cached compile.

### Tier-2 (code smells from the audit)

* **`_RestoreDtype` try/finally** — `_RestoreDtype.restore()` is
  now idempotent (`_restored` flag).  The dtype restoration in
  `design_optimize`'s scipy dispatch + final-eval block is
  wrapped in `try/finally` calling `restore()` explicitly.
  `__del__` retained as a safety net.  More robust under
  `KeyboardInterrupt` and exception unwinding.
* **`_merit_jac_auto` uses `scheme='forward'` + cached `f0`** —
  scipy already evaluates `merit_fn(x)` before calling jac on
  the same `x`.  `_merit_jac_auto` now passes that value as `f0`
  to `_fd_grad_for(... scheme='forward', f0=...)`.  FD eval
  count goes from `2N` to `N + 2` per gradient (~30% saving for
  N=10 free vars).  Threaded through internal helpers so
  callers that prefer the bit-identical central path still get
  it by default.
* **Cancellation protocol in `progress.py`** — new
  `CancellableProgress` class (exposed at `lumenairy.
  CancellableProgress`) with a `cancel()` method and
  `is_cancelled()` module helper.  Wired into all 4 scipy
  callbacks in `design_optimize` (`minimize`, `differential
  _evolution`, `basin_hopping`, `dual_annealing`).  When
  `cancel()` is called, the active scipy callback returns
  `True`; scipy stops gracefully and the post-loop final
  evaluation + `DesignResult` return still executes (so the
  caller gets the best-so-far state instead of a partial-data
  `KeyboardInterrupt`).
* **Merit propagator inconsistency warning** —
  `MultiWavelengthMerit`, `MultiFieldMerit`, and
  `ToleranceAwareMerit` all call `apply_real_lens` directly for
  off-nominal legs regardless of which `wave_propagator` was
  selected at `design_optimize` time.  Added a docstring caveat
  to each of the 3 classes AND a runtime `UserWarning` at
  `design_optimize` entry when `wave_propagator != 'real_lens'`
  and at least one of these Merit classes is in use.  Threading
  the propagator through sub-merit evaluations is a larger
  architectural change worth its own release.
* **Shared `_build_asm_H_square` helper** — the v4.13.0 Shack-
  Hartmann FFT batching introduced a local `_build_asm_H_for
  _lenslet` in `analysis/detector.py` that duplicated angular-
  spectrum H/bandlimit logic from `propagators/propagation.py`.
  Now consolidated: `_build_asm_H_square(N, dx, z, wavelength,
  dtype=None, bandlimit=True)` lives in `propagators/propagation
  .py` and is imported by `detector.py`.  Pinned at 1e-14 against
  a hand-built reference for both `bandlimit=True/False`,
  multiple grid sizes, complex64 dtype promotion, and the z=0
  unity short-circuit.  The propagator's own two inline H-build
  sites (NumPy chunked path and JAX functional path) stayed
  inline — they use cached freq-grid lookups + RAM-budgeted
  chunking (NumPy) and `xp.namespace` for JAX tracing, both of
  which materially differ from the helper's single-shot pure-
  NumPy build.
* **`_fd_grad_pure` `validate_f0` parameter** — when
  `scheme='forward'` and `f0` is supplied, `validate_f0=True`
  (default `False`) re-evaluates `f(x)` once and asserts
  `abs(f0 - f(x)) < tol * max(abs(f0), 1)`.  Caller contract
  documented in the docstring: stale `f0` silently produces
  wrong gradients without the validation gate.
* **BSDF TIS shape assertion** — `total_integrated_scatter`'s
  `np.broadcast_to(B, T.shape)` previously masked shape
  mismatches from subclass `BSDF.evaluate()` returns.  Now
  raises `ValueError` with expected vs actual shape on
  mismatch.

### Tier-3 cleanups

* **`memory.set_max_ram` validates non-negative input** —
  previously `set_max_ram(-5)` was silently accepted as
  -5 GB → negative bytes.  Now raises `ValueError`.
  `get_max_ram` added to `__all__`.
* **`MultiPrescriptionParameterization` duplicate detection** —
  duplicate `(prescription_index, *path)` entries in `free_vars`
  silently got separate `x[i]` slots competing for the same
  field.  Constructor now raises `ValueError` listing the
  duplicates.
* **JAX 0.4.20+ opaque PRNG keys** — `_is_jax_prng_key` now
  recognises opaque keys from `jax.random.key()` via the
  canonical `jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`
  check, with a `dtype.name.startswith('key<')` string fallback.
  Legacy uint32 / shape-trailing-2 typed keys still detected.
* **Dtype-aware zero in `apply_mirror` and `apply_aperture`** —
  the `xp.where(..., E, 0.0+0.0j)` literal is complex128 in
  Python.  On JAX with `x32` default this may upcast or fail
  jit.  Replaced with `xp.zeros((), dtype=E.dtype)` in both
  `apply_mirror` (audit-cited) and `apply_aperture` (same
  pattern in the same file, fixed conservatively).
* **`apply_mirror` aperture docstring** — said "ellipse" but the
  code computes a circle in physical coordinates.  Docstring
  corrected.
* **Stale pyc cleanup** — removed
  `tests/unit/__pycache__/test_audit_fixes_v4_13_0_perf_hfpi
  _bincount.cpython-314-pytest-9.0.3.pyc` (γ.1 revert
  leftover).

### New performance wins (beyond audit scope)

Open-ended perf survey across `propagators/` and `raytrace/` (the
modules not already optimised in v4.12.0 or v4.13.0) turned up
three wins:

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `vectorial_hfpi.accumulate_vector_to_grid` (1M paths, 256²) | 57.3 ms | 34.6 ms | **1.65-1.75×** |
| `analysis/detector.py:shack_hartmann` scatter-back (K=4096) | 2.97 ms | 0.12 ms | **9.5-25×** |
| `gbd.reconstruct_field_from_beamlets` (1024 beamlets, 96²) | ~1100 ms | ~870 ms | **1.2-1.5×** typical, up to 2.3× cache-warm |

* **`accumulate_vector_to_grid`** now shares the `ix`/`iy`/
  `inside`/`flat_idx` index arrays between the Ex and Ey scatter-
  adds (previously each call routed through `accumulate_to_grid`
  twice, recomputing the indices).  Bit-exact (`np.array_equal`).
  JAX path falls back to the original double-call form for
  tracing compatibility.
* **`shack_hartmann` scatter-back** replaced the `for k in
  range(K)` Python loop with vectorised fancy indexing using
  `iy_idx[ok]` / `ix_idx[ok]` / `cx_arr[ok]`.  Bit-exact.
* **GBD `reconstruct_field_from_beamlets`** fuses two `xp.exp`
  calls into one (`arg = -0.5*Q*rho2 + L*dX + M*dY`); replaces
  the `sum(a_b * phase, axis=-1)` reduction with `einsum('mnk,k
  -mn', phase, a_b)` to drop the `(Ny, Nx, chunk)` intermediate;
  switches accumulator to in-place `out +=` on NumPy/CuPy.
  Bit-near-exact (rel_err ~4e-16 vs the scalar reference).
  JAX rebind preserved.

**Deferred perf candidates** (catalogued for v4.14+):

* `propagators/hf.py:propagate_huygens_fresnel_with_opl_callable`
  per-pixel python loop calling `opl_fn` 16 times per output
  pixel (Van Vleck Hessian).  Could vectorise across `chunk
  _output` pixels.  Expected 5-20× on typical 256² grids with
  custom OPL.  Deferred: requires changing the documented
  callable contract (opl_fn must broadcast over arbitrary
  trailing shapes).
* Fused `(sag, dz_dh)` helper for `_intersect_surface` Newton
  iterations.  Currently `_surface_sag_xy` and
  `_surface_sag_derivatives_xy` both recompute `h = sqrt(x²+y²)`
  and re-traverse the aspheric polynomial dict on each of ~10
  Newton iters.  Expected 1.5-2× on aspheric surfaces.
  Deferred: requires reorganising `lenses.py:surface_sag
  _general`.
* Analytic pure-conic intersection extending the v4.12.1 Newton-
  skip from `kc==0` to all `kc` with no aspherics.  Expected
  5-10× on pure-conic surfaces.  Deferred: root selection for
  hyperbolic (`k<-1`) and degenerate paraboloid (`k=-1`,
  on-axis ray) cases needs broader validation.

### v4.13.0 CHANGELOG drift fixes (Tier-1 doc hygiene)

11 doc-vs-code mismatches in the v4.13.0 CHANGELOG entry below
have been retroactively corrected (line numbers, function names,
threshold directions, claim tightness).  The 12th (v4.12.2's
"6 clear-functions wired into lumenairy_context" should be 7 —
`clear_through_focus_scan_jax_cache` was the 7th but went
uncounted) is acknowledged here rather than retroactively edited
because v4.12.2 is already on PyPI.

### Test counts

* Pre-v4.13.1 baseline (v4.13.0 final state): 573 unit tests.
* v4.13.1 audit-response additions: 72 new tests across 8 new files
  (Agent 1: 14 in `test_audit_fixes_v4_13_0_jax_dtype_dy_siblings
  .py` (extended); Agent 2: 14 across `test_audit_fixes_v4_13_1
  _thin_grating_dock.py`, `_context_guards.py`, `_random_choice
  .py`; Agent 3: 26 in `test_audit_fixes_v4_13_1_agent3.py`;
  Agent 4: 18 across `_asm_h_helper.py` and 3 perf-pin files).
* Final: **654 unit tests passing**, **34/34 validation files
  passing**.

---

## [4.13.0] — 2026-05-17

**Bundle of three internal phases since v4.12.2 (PyPI-published):**

* Phase 1 — closes audit known-limitations S1, S2, S3 and L2, L3,
  L4, L6, L8 from `docs/audits/AUDIT_V4_12_1_2026_05_16.md` (storage dtype
  preservation, codegen aperture-stop + wavelength sentinel, ghost
  R/r convention, JAX dtype + `jax_enable_x64`, `PropagationResult.dy`
  + `Source.dy`, sibling mirror-guards, `apply_mirror` xp + dy,
  zarr thread-safety).
* Phase 2 — sweeps `except Exception:` clauses across the non-UI
  codebase from 99 → 3 justified sites (typed exceptions
  everywhere else, three WARN-BEFORE-PASS upgrades).
* Phase 3 — Tier-2 perf wins from the same audit: 10× thin-grating,
  188× BSDF TIS, 4.43× Chebyshev freeform, 3.26× coating-stack,
  10.8× Shack-Hartmann FFT batching, 4–72× Seidel field sweep,
  smaller wins on `wave_opd_2d` and `_fd_grad`.  Also: rcwa.py →
  thin_grating.py file rename with no back-compat shim (sole-user
  waiver).

All 573 unit tests pass; 34/34 validation files pass.

### Breaking changes

* **`rcwa.py` → `thin_grating.py` file rename, no back-compat
  shim.**  `import lumenairy.elements.rcwa` raises
  `ModuleNotFoundError`.  `import lumenairy.ui.rcwa_dock` likewise.
  The public symbols (`thin_grating_efficiency_1d`,
  `grating_efficiency_vs_wavelength`) were already renamed in
  v4.4.0; v4.13.0 finishes the rename at the file level.
* **`wavelength` is now required by `io.codegen
  ._decompose_prescription`.**  Previously a missing `wavelength`
  silently defaulted to `1.31e-6` (NIR).  Now raises `ValueError`
  with a helpful message naming the parameter.  Visible-band
  Zemax imports that relied on the silent NIR default must now
  pass `wavelength` explicitly.

### Phase 1 — Audit known-limitations closure

**S1 — `io.storage` complex-dtype preservation through append.**
`save_jones_field_h5`, `append_plane_h5`, `_zarr_append_plane`, and
the unified `append_plane` dispatcher gained a `preserve_dtype`
parameter that is threaded down through the write path.  Previously
the append-side stack silently promoted complex64 inputs to
complex128 on every plane after the first.  Default behaviour is
preserved: callers that omit `preserve_dtype` continue to see the
v4.12.x promotion.

**S2 — `io.codegen` aperture-stop emission + wavelength sentinel.**
`_decompose_prescription` now emits an `{'type': 'aperture', ...}`
step before mirrors / dummy planes / real-lens groups whenever
`is_stop=True` on the source surface.  The Zemax loader threads the
`is_stop` flag through `prescriptions.py` so the stop reaches
codegen with its identity intact.  Separately, the silent default of
`wavelength_nm = 1.31e-6` was replaced with a `ValueError`; callers
that previously got NIR by accident now get a clear failure at
codegen time.

**S3 — `analysis.ghost` `R` vs `r` convention disambiguation.**
A top-of-module convention block now disambiguates uppercase `R`
(Fresnel reflectance) from lowercase `r` (curvature radius).
Local variables renamed accordingly (`R_i_val → r_i`,
`R_j_val → r_j`).  Public dict keys are unchanged for back-compat
consumers.

**L2 — JAX complex-dtype routing.**
Added `_resolve_jax_complex_dtype()` and `_resolve_jax_real_dtype()`
helpers in `propagators/propagation.py`.  When complex128 is
required, the helper auto-enables `jax_enable_x64` and emits a
`RuntimeWarning` rather than silently truncating.  The warning is
implicitly one-shot (gated by the `jax_enable_x64` state flip after
the first call), not enforced via `warnings.simplefilter('once')`.
Threaded through
`system.py:propagate_system_jax`, `_lens_jax.py` (three call
sites), and `analysis/phase_retrieval.py`.  The
`_PROPAGATE_SYSTEM_JAX_CACHE` key now includes `str(np.dtype(cdtype))`
so a float32-then-float64 sweep stops aliasing onto a single XLA
binary.

**L3 — `PropagationResult.dy` and `Source.dy` fields.**
`PropagationResult` gained a `dy` field (defaults to `dx` in
`__post_init__`); `dy_out` / `dx_out` aliases preserved.
`_coerce_field` in `propagators/dispatch.py` now returns
`(field, dx_out, dy_out)`; threading carried through every internal
caller.  `Source` (in `sources/core.py`) gained a matching `dy` that
`to_source()` forwards.  Anamorphic-grid propagation can now round-
trip metadata without losing the second axis.

**L4 — Sibling mirror-guards in `apply_real_lens_*` variants.**
`apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`, and
`lenses_maslov.apply_real_lens_maslov` all gained the pre-flight
`is_mirror=True` guard previously only present in
`apply_real_lens`.  Calls into a folded prescription that should
have been split now raise `ValueError` consistently across the
trio + maslov.

* L4b — `phase_retrieval.py` raises `NotImplementedError` when an
  `initial_guess` is passed on the JAX backend (the JAX path does
  not currently honour it); migration message included.
* L4c — `phase_retrieval.py` warns + synthesises an empty history
  list when `return_history=True` is requested on the JAX backend.

**L6 — `apply_mirror` xp dispatch + `dy` parameter.**
`elements/elements.py:apply_mirror` now resolves the backend
namespace via `_xp_of(E_in)` and accepts a `dy` parameter for
anamorphic grids.  NumPy short-circuit retained for the
numba-aspheric fast path; JAX and CuPy take the xp-native inline-sag
branch.  Backwards compatible: `dy=None` reproduces the v4.12.x
square-grid behaviour exactly.

**L8 — Zarr thread-safety guard.**
Added a module-level `_ZARR_MKDIR_PATCH_LOCK = threading.Lock()` in
`io/storage.py` to guard the `Path.mkdir` monkey-patch in
`_open_zarr_group_safe`.  Concurrent writers no longer race on
patch-install vs patch-restore.

### Phase 2 — `except Exception:` sweep

Non-UI library files contained 99 `except Exception:` clauses
(audit's "242" figure swept the full repo including tests,
validation, and commentary; the real non-UI count was 99).  After
the sweep:

* 3 KEEP-AS-IS, justified inline:
  * `_context.py:299` — atexit handler tolerating module-level
    global teardown.
  * `optimize/core.py:2683` — `_RestoreDtype.__del__` cleanup
    tolerating shutdown.
  * `propagators/hfpi.py:84` — optional-dep guard around
    `import jax`.
* ~85 narrowed to typed tuples covering the documented failure
  modes (e.g. `(RuntimeError, MemoryError, ValueError, TypeError,
  AttributeError)` for pyFFTW failures; `(ImportError, RuntimeError,
  AttributeError)` for cache-clear guards; `(ValueError,
  RuntimeError, ZeroDivisionError, np.linalg.LinAlgError,
  IndexError)` for `system_abcd` fallbacks; etc.).
* 3 WARN-BEFORE-PASS upgrades — surfaces in failure paths that
  would previously have silently degraded:
  * `analysis/field.py petzval_radius` — a missing glass entry was
    returning NaN, which downstream could be confused with "the
    field is perfectly flat."  Now warns explicitly with the glass
    name + wavelength.
  * `propagators/hf.py` LG decomposition fallback — falling back to
    a single `(p=0, l=0)` plane-wave mode makes the asymptotic
    propagator essentially useless; failure is now surfaced.
  * `optimize/core.py design_optimize plane_logger` callback — was
    silently swallowing all logger failures for the duration of an
    optimization run.  Promoted to WARN-BEFORE-PASS so users see
    telemetry-callback bugs immediately instead of an empty log
    file at the end.

Pinning test:
`tests/unit/test_audit_fixes_v4_13_0_except_sweep.py` with 6 tests
including a regression budget guard (non-UI `except Exception:` count
≤ 15), two behavioural pins (`petzval_radius` warns, narrow tuples
drop on expected failures), and one source-string pin via
`inspect.getsource` for the `design_optimize plane_logger` warning
(behavioural exercise was unreliable across scipy versions).

### Phase 3 — Tier-2 performance wins

Three disjoint-scope agent groups (elements/UI, analysis/optimize,
propagators/raytrace) executed in parallel.  Measured speedups
(`time.perf_counter`, representative workloads on Win11 / Python
3.14):

#### Group α — elements + UI

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `thin_grating_efficiency_1d` (n_orders=25) | 198 us | 18.9 us | **10.5×** |
| `bsdf.total_integrated_scatter` (256×128) | 384 ms | 2.05 ms | **188×** |
| coating-stack matmul chain (50 layers) | 0.48 ms | 0.15 ms | **3.26×** |
| freeform Chebyshev (64×64, 16 coeffs) | 1.63 ms | 0.37 ms | **4.43×** |

**`elements/rcwa.py` → `elements/thin_grating.py` rename.**  The
file name "rcwa" was misleading: the function inside is the
analytical scalar thin-phase grating formula, not rigorous coupled-
wave analysis.  The functions themselves were renamed in v4.4.0
(`thin_grating_efficiency_1d`, `grating_efficiency_vs_wavelength`);
v4.13.0 finishes the rename at the file level:

* `lumenairy/elements/rcwa.py` → `lumenairy/elements/thin_grating.py`
* `lumenairy/ui/rcwa_dock.py` → `lumenairy/ui/thin_grating_dock.py`
  (class `RCWADock` → `ThinGratingDock`)
* `lumenairy/__init__.py` import path + Tier-7 comment updated.
* `lumenairy/elements/__init__.py` module docstring updated.
* `lumenairy/ui/main_window.py` (7 token occurrences across 6
  logical sites: import, widget construct, dock construct, dock-key,
  menu label, show_and_raise lambda, visible-list) +
  `lumenairy/ui/workspace.py` (dock-key mapping) updated.

**No back-compat shim** is installed at either old path
(`import lumenairy.elements.rcwa` raises `ModuleNotFoundError`).
This is per the explicit waiver from the sole user of the library.

**`bsdf.total_integrated_scatter`** is now a fully-vectorised 2D
meshgrid + single `integrand.sum() * dθ * dφ` reduction (rectangle
rule on a regular θ-φ grid) rather than a per-(θ_i, θ_s) inner-
product loop.  This is the biggest single-call speedup in the
v4.13.0 batch.

**Coating-stack matmul** moved from a Python `M = M @ M_layer` loop
to a tournament reduction over a `(N, 2, 2)` per-layer tensor.  The
agent also switched the inner Snell-chain math from `np.sin/arcsin`
to `math.sin/asin` for scalar wavelengths — bit-near-identical via
libm (pinned at `atol=1e-12` rather than ULP-exact since libm
implementations can diverge in the last few bits across versions) —
lifting the win from 1.5× into the 3-5× target band.

**Freeform Chebyshev** hoists the `arccos(rho)` out of the per-order
loop and caches the `T_i(rho)` factors keyed by polynomial order,
exploiting the fact that many `(i, j)` coefficient pairs share an
`i` or `j` so 2N cos calls collapse to roughly `2 sqrt(N_coeffs)`.

#### Group β — analysis + optimize

| Hot path | Old | New | Speedup |
|---|---|---|---|
| Shack-Hartmann WFS (16² lenslets, 256² grid) | 72.9 ms | 6.76 ms | **10.8×** |
| `wave_opd_2d` (512²) | 52.5 ms | 27.7 ms | **1.89×** |
| `_fd_grad_pure` (N=20 quadratic, forward) | 0.159 ms | 0.071 ms | **2.23×** |

**Shack-Hartmann FFT batching.**  `analysis/detector.py:
shack_hartmann` previously ran two nested per-lenslet double-loops
(reference + measurement), each computing an `np.fft.fft2` on a
single sub-aperture.  The reference path is now a single fft2 on a
stacked `(n_lenslets², sa_pixels, sa_pixels)` 3D array; the
measurement path is the same.  A new `_build_asm_H_for_lenslet`
helper inside `detector.py` duplicates a thin slice of the
angular-spectrum H-build logic for the sub-aperture geometry,
avoiding a scope-crossing edit into `propagators/` (a follow-up
in v4.13.1 consolidates this into a shared
`_build_asm_H_square` helper).  NaN sentinels for OOB lenslets
and the `sa_pixels >= 2` (i.e. `pitch >= 2*dx`) sampling guard
are preserved.  Note: this guard raises `ValueError`, not a
warning.

**`wave_opd_2d` axis unwrap.**  `analysis/core.py:wave_opd_2d`
replaces the row-then-column per-pixel Python unwrap loop with
`np.unwrap(opd, axis=1)` then `np.unwrap(..., axis=0)` (matching
the legacy row-first traversal order).  At N=512
the Python iteration overhead was about half the run time; the
inner unwrap step itself was already compiled, so the 1.89× win is
real but below the speculative 5-10× target.  Correctness preserved
to 1e-12 on smooth quadratics, twice-wrapping tilts, and masked
NaN regions.

**`_fd_grad_pure` / `_fd_grad_for` central-vs-forward scheme.**
Pre-v4.13 the helper used central differences (2N evals,
`f(x ± h*e_i)`).  The audit's request to "reuse the centre value"
was based on a forward-FD model — central FD has no `f(x)` to
reuse, so the perf win required switching the scheme.  v4.13.0
parameterises this:

* `_fd_grad_pure(...)` and `_fd_grad_for(...)` accept
  `scheme='central'|'forward'`, default `'central'`.
* `scheme='central'` (the default) preserves bit-identical
  gradient values with pre-v4.13 behaviour at 2N evals,
  O(h²) truncation.
* `scheme='forward'` is the opt-in perf path (N+1 evals, or N with
  the optional `f0=<known value>`).  O(h) truncation.

`design_optimize._merit_jac_auto` keeps the default `'central'`
explicitly, so no existing optimisation run sees a behavioural
change.  Callers that prefer speed can opt into `'forward'` at the
helper level.

#### Group γ — propagators + raytrace

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `seidel_field_sweep` (5 fields, singlet) | (ref) | (ref/4) | **4.2×** |
| `seidel_field_sweep` (50 fields, singlet) | (ref) | (ref/37) | **37.5×** |
| `seidel_field_sweep` (100 fields, singlet) | (ref) | (ref/72) | **72.0×** |

**`raytrace/seidel_analysis.py:seidel_field_sweep` per-field hoist.**
The paraxial Seidel formalism is exactly linear in the chief-ray
initial conditions, and those scale linearly with field angle.  The
sweep now does a single `seidel_coefficients(... field_angle=1.0)`
call and applies the analytical per-field scaling (`S1, S4 ∝ 1`;
`S2, y_chief ∝ σ`; `S3 ∝ σ²`; `S5 ∝ σ³`).  All field-independent
work — glass-index lookups, pre-stop ABCD, marginal-ray trace, full
`system_abcd` — is hoisted out of the loop.  Element-by-element
agreement with the pre-hoist reference is well below the test
pin tolerance of `< 1e-12 absolute` (numerics on the singlet
test case land in the 1e-15..1e-20 range in practice).

**γ.1 — HFPI bincount swap reverted before ship.**  The initial
agent run replaced the `np.add.at(out, flat_idx, w_masked)` scatter
in `propagators/hfpi.py:accumulate_to_grid` with a `np.bincount`-
based path on the premise that `add.at` was a Python-level loop.
That premise pre-dated NumPy 1.25; NumPy ≥ 1.25 has vectorised
`add.at` for complex via `_PyArray_UFuncBufferedAtVectorized`.
Measured speedup on the actually-shipping NumPy was 0.4–1.0×
(a wash, leaning regression), so the patch was reverted.  No
behavioural change vs v4.12.2 in this path.

### Bug-fix and discipline notes

* Phase 2 audit count discrepancy resolved.  Audit reported 346
  `except Exception:` clauses; the real non-UI count was 99 (the
  audit swept tests, validation, and prose into the same `grep`).
* Phase 3 Group α / γ race observed mid-run: Group γ briefly
  imported a non-existent `elements/rcwa.py` reference while
  Group α was mid-rename.  Resolved by Group α's completion; no
  artefact in shipped code.
* `tests/unit/test_audit_fixes_v4_12_0_round4_dispatch.py` —
  `test_coerce_field_unpacks_tuple` was rewritten in-place using
  `result[0]` / `result[1]` subscript indexing (no `pytest.mark.skip`)
  because L3 changed `_coerce_field`'s signature to a 3-tuple
  `(field, dx_out, dy_out)`.  All 7 tests in the class collect and
  pass.  The dispatcher-level 3-tuple pin is in the v4.13.0
  dy-siblings test file.

### Test counts

* Pre-Phase-3 baseline (Phase 1 + 2 landed): 536 unit tests.
* Phase 3 additions: 12 perf pins (α) + 12 (β) + 7 (γ) = 31, minus
  6 bincount tests deleted with the γ.1 revert + net 0 from the
  fd_grad test rewrite.  Final: **573 unit tests passing**,
  **34/34 validation files passing**.

## [4.12.2] — 2026-05-17

**Closes the round-5 / v4.12.1 pre-PyPI audit blockers**
(`docs/audits/AUDIT_V4_12_1_2026_05_16.md`).  Three documentation-drift items
become true (NumPy `through_focus_scan` H-hoist actually
implemented, `through_focus_scan_jax` JIT cache actually
implemented, 878× benchmark headline reconciled to ~300×).  Cache
hygiene infrastructure landed: `clear_asm_caches()` extended,
unbounded jit caches converted to LRU(32), 4 new public
`clear_*_cache()` helpers, all wired into
`lumenairy_context(clear_caches_on_exit=True)`.  Two test pins
strengthened, one coverage-test made bug-exercising, one benchmark
stub replaced.  All 482 unit tests pass; full validation suite (34
files / 314 tests) passes.

### A1 / A2 — Build & test configuration (`pyproject.toml`)

* `pytest-benchmark>=4.0` added to `[project.optional-dependencies]
  .dev`.  Round-5 verified the release notes had claimed this since
  v4.12.0 but the `dev` extra never gained it.  `pip install
  lumenairy[dev]` now actually installs pytest-benchmark.
* `bench` pytest marker registered in
  `[tool.pytest.ini_options].markers`.  `benchmarks/conftest.py`
  applies this marker to every collected benchmark; with
  `--strict-markers` set, an unregistered marker fails collection.
  Now collects cleanly.

### A3 — Cache-clear / FFT-toggle symbols exposed at top level

These existed in submodules but were not importable from
`lumenairy`:

* `set_fft_auto_promote`, `get_fft_auto_promote`
* `clear_zernike_basis_cache`
* `clear_lg_polynomial_cache`
* `clear_trace_jax_cache` (NEW)
* `clear_through_focus_scan_jax_cache` (NEW)
* `clear_propagate_system_jax_cache` (NEW)
* `clear_phase_retrieval_caches` (NEW)

All exported via `lumenairy/__init__.py` and listed in `__all__`.
`la.set_fft_auto_promote(False)` and the six `la.clear_*_cache()`
helpers now work.

### A5 — Cache hygiene infrastructure

* `clear_asm_caches()` extended to also clear `_PYFFTW_PLAN_CACHE`
  and `_PYFFTW_BAD_SHAPES` -- the pyFFTW double-buffer + auto-
  promote work added in v4.12.0 now actually gets released when
  callers ask for a fresh state.
* Unbounded jit caches converted to `OrderedDict` + LRU
  (`maxsize=32`).  Round-5 L1 + v4.12.1 cache-regression flagged
  that `_PROPAGATE_SYSTEM_JAX_CACHE` (v4.12.0) and `_TRACE_JAX_CACHE`
  (v4.12.1) were plain `Dict[Any, Any]` -- an optimizer iterating
  over different prescriptions accumulated a compiled XLA binary
  per iteration with no eviction.  Both converted; phase-retrieval
  kernel caches given the same treatment.
* `lumenairy_context(clear_caches_on_exit=True)` now calls all
  six `clear_*` functions (each guarded with `try/except` so a
  missing optional dependency, e.g. JAX, doesn't break context
  exit).

### D3 — NumPy `through_focus_scan` H-hoist actually implemented

v4.12.0 release notes claimed `through_focus_scan` (NumPy backend)
hoisted the input FFT, kx/ky, propagating mask, and target dtype
outside the z-loop.  Source code did not.  v4.12.0's 4.7× speedup
came from the underlying pyFFTW MEASURE auto-promote inherited via
`angular_spectrum_propagate`, not from any per-z hoisting.

v4.12.2 makes the claim true.  `through_focus_scan` (NumPy) now
precomputes `E_fft_shifted`, `kz_safe`, `propagating`, and the
target complex dtype once before the loop.  Per-z work reduces to
`H_z = where(propagating, exp(1j*kz_safe*z), 0) * bl_mask_z` then
`E_z = fftshift(ifft2(ifftshift(E_fft_shifted * H_z)))`.  Band-
limit masks remain per-z (cached via the existing
`_get_or_make_bandlimit`).

Pinning tests
(`tests/unit/test_perf_v4_12_0_through_focus.py::
TestThroughFocusScanNumPyHoistActuallyHoists`):
* `test_fft2_called_only_once_across_scan` -- mocks `_fft2` and
  asserts `call_count == 1` (was `n_z` pre-fix).
* `test_z_invariant_caches_built_once` -- mocks
  `_get_or_make_freq_grids` and asserts `call_count == 1`.
* All 13 pre-existing bit-near-exact pins still pass.

### D4 — `through_focus_scan_jax` JIT cache actually implemented

v4.12.0 release notes claimed `through_focus_scan_jax` jit-caches
the inner ASM kernel.  Source code had no `@jax.jit` wrap and no
module-scope cache; every Python call re-traced.

v4.12.2 makes the claim true.  Factored the inner kernel out of
the closure into `_build_through_focus_scan_jax_kernel(Ny, Nx, dx,
wavelength, bandlimit, dtype_str)` which wraps `jax.vmap(_asm_one
_z, in_axes=(None, 0))` with `jax.jit`.  Module-scope
`_THROUGH_FOCUS_SCAN_JAX_CACHE: OrderedDict` (LRU `maxsize=32`)
caches the compiled kernel per signature.  `clear_through_focus
_scan_jax_cache()` exported.

Benchmark (`benchmarks/test_bench_jax_jit.py`):
* N=64, 7 z-planes, complex128: first **153 ms** -> warm **1.97
  ms** = **~77×**.
* N=128, 7 z-planes, complex128: first **128 ms** -> warm **4.95
  ms** = **~26×**.

### 878× → ~300× warm-call `trace_jax` benchmark reconciliation

v4.12.1 release notes claimed `trace_jax` warm-call: **878×**
(127 ms → 0.40 ms).  Those numbers don't reconcile (`127/0.40 =
317×`, not 878×).  Fresh measurement on a stable system state
(1001-ray AC254-100-C-equivalent doublet, median of 20 warm calls,
5 passes): first **140 ms**, warm **0.47 ms**, speedup **~300×**.
Updated in `CHANGELOG.md`, `README.md`,
`.release_notes_v4.12.1.md`, and the wiki.  Regression pin
threshold tightened from `>= 100×` to `>= 200×`.

### Item 11 — Coverage test for Zemax coord-break STOP marker tightened

The v4.12.1 test placed the coord-break AFTER `stop_surface=1`,
which doesn't exercise the pre-v4.11.2 off-by-one bug.  New test
`test_coord_break_at_index_0_does_not_bump_stop` places the
coord-break at `surf_num=1` BEFORE the stop, exports + re-loads,
asserts STOP lands on SURF 3 (second refractive), not SURF 1
(the COORDBRK).  Pre-fix would have emitted STOP on SURF 1.

### Replaced benchmark stub

`benchmarks/test_bench_jax_jit.py` had a `trace_jax — deferred to
v4.12.1` stub.  Replaced with a real `test_bench_trace_jax_first
_vs_warm` benchmark.

## Known limitations (deferred to v4.13 / v4.14)

Audit `docs/audits/AUDIT_V4_12_1_2026_05_16.md` identified items below as
non-blocking for the v4.12.x line.

### Silent-data-loss class (S1-S3)

* **S1 — `io/storage.py` append-side hardcodes `complex128`.**
  Lines 282-283 and 342 use `np.asarray(field, dtype=np.complex128)`
  unconditionally.  Single-shot save APIs honour `preserve_dtype`;
  the append APIs (used by `MhsPipeline.run(store=...)` and
  `replay_run`) do not.  A complex64 simulation streamed to disk
  via the append path silently doubles its on-disk size.
* **S2 — `io/codegen.py` aperture-stop drop + 1.31 µm wavelength
  default.**  Zemax-to-Python codegen silently drops aperture-stop
  surfaces during `_decompose_prescription` and defaults to
  `wavelength = 1.31e-6` (1310 nm NIR) when none supplied.
* **S3 — `analysis/ghost.py` `R_i`/`R_j` convention conflict.**
  `focus_z_estimate` docstring uses `|R_i|, |R_j|` for curvature
  radii while `R_i`, `R_j` elsewhere in the module denote Fresnel
  reflectance.

### Structural / latent (L1-L8)

* **L2 — JAX path `complex64` hard-casts.**  `system.py:833` and
  similar sites silently override `set_default_complex_dtype(np
  .complex128)`.  JIT cache key does not include dtype.  Cross-
  module inconsistent (`apply_real_lens_traced_jax` reads
  `jax.config.jax_enable_x64`).
* **L3 — `PropagationResult` missing `dy` field.**  `_coerce_field`
  extracts `dx_out` from tuple-returning kernels but discards
  `dy_out`.  Benign for `dx == dy`; lossy for anamorphic Fresnel.
* **L4 — Sibling-gap remnants.**  Mirror-guard not applied to
  `apply_real_lens_maslov`, `apply_real_lens_traced_jax`,
  `apply_real_lens_maslov_jax`.  `error_reduction(backend='jax')`
  and `hybrid_input_output(backend='jax')` dispatch don't forward
  `initial_guess`.  `gerchberg_saxton(backend='jax',
  return_history=True)` silently drops `return_history`.
* **L5 — 346 `except Exception:` clauses remain in core scientific
  code.**  Many `pass`/return-NaN without logging.
* **L6 — `apply_mirror` doesn't use `array_namespace` dispatch.**
  Every other `apply_*` in `elements.py` switched in 4.10/4.11.
  CuPy/JAX inputs fall through to a NumPy code path silently.
  Also missing `dy` parameter.
* **L8 — `_open_zarr_group_safe` non-thread-safe `Path.mkdir`
  monkey-patch.**  Two threads racing through `append_plane_h5`
  can leave the monkey-patch in an inconsistent state.

L1 and L7 are now resolved (jit cache eviction in A5; through_focus
scan_jax cache exists post-D4 so the benchmark hygiene concern
applies).

## [4.12.1] — 2026-05-16

**Closes the three perf items deferred from v4.12.0, adds the
14 missing regression tests round-4 audit identified, and lands
the B1-10 half-pixel grid drift fix.**  All 453 unit tests pass;
full validation suite (34 files / 314 tests) passes.

### Performance wins recovered from v4.12.0 deferrals

* **`trace_jax` jit cache via pytree-registered prescription wrapper**
  -- ~300x warm-call speedup (140 ms cold -> 0.47 ms steady-state,
  measured on a 1001-ray AC254-100-C-equivalent doublet, median of
  20 warm calls).  The pre-PyPI release note quoted 878x with
  inconsistent absolute timings (127 ms / 0.40 ms = 317x); v4.12.2
  reconciles to one consistent set after re-running on a stable
  system state.
  v4.12.0 attempted this with a flat-tuple cache key and reverted
  because `jax.grad(fit_canonical_polynomials_jax)` returned NaN.
  Root-cause investigation found the NaN was not the cache key
  but a JAX bug: `jax.jit` wrap + downstream `jnp.linalg.lstsq`
  backward produces NaN in `dot_general` on near-rank-deficient
  matrices (the canonical-poly 4-D Chebyshev basis triggers it).
  v4.12.1 fix: `JaxPrescription` pytree wrapper class for clean
  cache mechanics, plus `_running_under_trace` guard that
  bypasses the jit-cache layer whenever any pytree leaf is a
  `jax.core.Tracer`.  Under `jax.grad`/`jax.jit`/`jax.vmap` the
  calling transform owns the trace and the extra jit wrap is
  redundant; bypassing preserves v4.11.2 grad semantics.  Pin:
  `jax.grad through fit_canonical_polynomials_jax is finite`
  passes with `grad=1.0274e+04` (was NaN in v4.12.0).

* **Raytrace Newton spherical fast-path** -- 1.50x speedup on
  1k-ray doublet trace (735 us -> 491 us).  v4.12.0 attempted
  this with BOTH (a) Newton skip AND (b) analytic spherical
  normal `(x/R, y/R, (z-R)/R)`; the analytic normal compounded
  a 1.17e-3 rel error through the Maslov asymptotic chain.
  v4.12.1 ships only (a) -- pure-spherical surfaces
  (`conic == 0`, no aspheric/biconic/freeform, finite radius)
  skip Newton via the analytic quadratic root; `_refract`/
  `_reflect` keep using the numerical-radial-derivative-based
  `_surface_normal` so LSB rounding is bit-identical to v4.11.2.
  Smaller speedup (1.50x vs the 1.64x v4.12.0 attempted) but
  the cross-backend correctness pin
  `aberration_tensor_lg00_jax matches NumPy` now passes with
  rel_err = 4.53e-04 (was 1.17e-3 broken; 4.83e-4 baseline).

* **B1-10 half-pixel grid drift unified** -- five propagator
  files (`gbd.py`, `mhs.py` x2 sites, `subaperture.py`,
  `optimize/core.py`) switched from `(arange(N) - N/2 + 0.5)*dx`
  cell-centred to `(arange(N) - N/2)*dx` pixel-centred so they
  match the library-wide ASM / Fresnel / RS / sources
  convention.  GBD self-roundtrip half-pixel walk-off
  eliminated; ASM <-> GBD centroid agreement now within
  `0.1*dx` (was `0.5*dx`).  Legitimate `+0.5` usages
  (Chebyshev nodes, DOE diffraction orders, hardware detector
  pixel centres) confirmed and left untouched.

### Test coverage gaps closed

Round-4 audit identified 14 v4.11.2 fixes that landed in code
but lacked regression pins.  v4.12.1 closes all 14 in
`tests/unit/test_audit_fixes_v4_12_1_coverage.py` (21 new tests
across 14 classes):

* `compute_psf` non-square pupil error
* `apply_detector` non-integer pixel ratio area scaling
* `find_best_focus` all-NaN guard
* `monte_carlo_tolerancing_linearized` `a_k >= 0` clamp
* `load_material` RuntimeWarning on dropped dispersion
* `Source.*` `**factory_kwargs` propagation
* `apply_real_lens_traced` M_x / M_y transpose
* NaN sentinel mask in `apply_real_lens`
* `stop_index != 0` warns in `_traced` / `_maslov` paths
* Freeform-terms `RuntimeWarning` in thin-element
* Zemax coord-break STOP marker (was only mirror tested)
* JAX <-> NumPy phase-retrieval cross-backend parity
* Cassegrain S1 / S2 / S3 / S5 hand-derivation (S4 was already
  hand-pinned in v4.11.2; now all five totals are pinned)
* Richards-Wolf first-null vs paraxial Airy at low NA

### Weak tests strengthened

* `test_real_E_in_yields_complex_out` -- replaced
  `inspect.getsource` source-string scanning with a behavioural
  pin: pass a real `E_in` through the HF Chebyshev quadrature,
  assert `out.dtype` is complex and `out.imag` RMS > 0.
* `test_axial_opl_path_does_not_emit_failure_warning` --
  retained as smoke test; added `test_axial_opl_is_actually
  _non_zero` which monkey-patches `apply_abcd_to_beamlets` to
  capture the `axial_opl` kwarg and asserts it is finite and
  > 1 mm (matches `n_BK7 * 2 mm thickness` for the test
  singlet).

### Test harness cleanup

* `validation/io/test_io.py:196` bare `except: return True,
  'skipped'` removed.  The exporter+loader contract has been
  round-trippable since v3.7.0; a raise from `load_zemax_zmx`
  on an exported file is a real regression and should fail
  loudly.

### Validation: 34/34 files / 314 tests pass

Including the two cross-backend pins that caught the v4.12.0
regressions:
* `aberration_tensor_lg00_jax matches NumPy` -- rel_err = 4.53e-04
* `jax.grad through fit_canonical_polynomials_jax is finite` --
  grad = 1.0274e+04

### Tooling

* `JaxPrescription` pytree wrapper now exported from
  `lumenairy.raytrace.jax_trace` -- users who want to benefit
  from the warm-call cache can build a JaxPrescription once
  and pass it into `trace_jax` directly.
* All new pinning tests added to `tests/unit/` follow the
  v4.11.2+ naming convention
  (`test_audit_fixes_v4_12_1_*.py`).

## [4.12.0] — 2026-05-16

**Combined performance + round-4 pre-PyPI audit response.**  v4.12.0
bundles two parallel work streams: (1) a Tier-1 performance pass
that shipped ~10x speedups on hot paths, and (2) a round-4 pre-PyPI
audit that closed ~20 release blockers identified in
`AUDIT_ROUND4_2026_05_16.md`.  All 390 unit tests pass; the full
validation suite (34 files / 314 tests) passes.

### Performance — Tier-1 wins (vs v4.11.2 baseline)

| Workload | v4.11.2 | v4.12.0 | Speedup |
|---|---|---|---|
| ASM propagate 1024^2 complex128 | 165 ms | 39 ms | **4.3x** |
| `apply_real_lens` 4-surf 512^2 | 140 ms | 61 ms | **2.3x** |
| `through_focus_scan` 7-pt N=256 | 162 ms | 35 ms | **4.7x** |
| `through_focus_scan` 31-pt N=256 | ~715 ms | 149 ms | **4.8x** |
| `propagate_through_system_jax` warm | 148 ms | 0.91 ms | **163x** |
| `gerchberg_saxton_jax` 50-iter warm | 454 ms | 12.5 ms | **36x** |
| `error_reduction_jax` 50-iter warm | 256 ms | 5.6 ms | **46x** |
| `propagate_hf_chebyshev_quadrature` 32^2 chunk=1024 | 5575 ms | 255 ms | **21.8x** |
| `lg_polynomial` warm cache (LG_{3,2}) | -- | 21x faster | **21x** |
| `zernike_basis_matrix` warm hit | 22 ms | 1.8 us | **~12000x** |
| `zernike_decompose` 10-call loop | 298 ms | 80 ms | **3.7x** |

The 4.7x `through_focus_scan` speedup propagates through
`MultiWavelengthMerit` / `MultiFieldMerit` / Monte-Carlo
tolerancing:  100-trial MC at 31-pt N=256 drops from **71 s -> 15 s**.

Implementation:
* **pyFFTW double-buffer plan cache** + auto-promote
  `FFTW_ESTIMATE -> FFTW_MEASURE` at the 5th call (one-shot per
  cache key).  Saves 256 MB-1 GB allocation per call on large
  complex128 grids; `set_fft_auto_promote(False)` disables for
  startup-sensitive workflows.
* **`through_focus_scan` (NumPy) H-hoist** outside the z-loop --
  per-z work reduces to `H = exp(1j*kz*z) * propagating *
  bandlimit_z` + one `ifft2`.  Mirrors the JAX twin's pre-existing
  structure.  Bit-near-exact (abs err 0.0) vs per-z reference.
* **`zernike_basis_matrix` content-fingerprint cache** -- every
  Zernike merit in a CompositeMerit eval and every FD Jacobian
  column hits the cache.  `clear_zernike_basis_cache()` exposed
  for in-place-mutation escape.
* **JAX jit caches** at `propagate_through_system_jax`,
  `gerchberg_saxton_jax` / `error_reduction_jax` /
  `hybrid_input_output_jax`, and the inner ASM kernel inside
  `through_focus_scan_jax`.  Module-scope `OrderedDict` caches
  keyed on element-chain signature or `n_iter`.
* **`propagate_modal_asymptotic` `lg_polynomial` hoist** +
  `lru_cache(maxsize=256)` -- moves the LG-polynomial build
  outside the per-pixel Newton loop; per-pixel work drops from
  `N_pixels * N_modes` to `N_modes` recomputes.
* **`propagate_hf_chebyshev_quadrature` chunk vectorisation** --
  replaces scalar pixel loop with `np.einsum('cyx,yx->c', kernel,
  E_in)`.  New `max_chunk_memory_mb=256.0` kwarg caps peak
  alloc; effective chunk auto-shrinks if requested chunk would
  overshoot.  Fixes a pre-existing latent shape-mismatch bug
  along the way.

Two perf items were reverted before shipping (their isolated
correctness pins passed but the cross-suite validation caught
cross-backend regressions):
* Raytrace Newton spherical fast-path + analytic-normal stash
  caused a 1.17e-3 NumPy<->JAX drift in `aberration_tensor_lg00
  _jax`.  Deferred to v4.12.1 with stricter cross-backend pin.
* `trace_jax` flat-tuple jit cache broke
  `jax.grad(fit_canonical_polynomials_jax)` (returned NaN).
  Deferred to v4.12.1 with pytree-registered prescription
  wrapper.

### Round-4 pre-PyPI audit fixes

**B0 — User-facing showstoppers (Tier 0)**

* **B0-1 README cookbook examples fixed** -- 11 broken code
  blocks (positional `apply_real_lens` calls now keyword;
  `create_gaussian_beam` missing `wavelength` added; renamed
  `load_zmx_prescription` updated to `load_zemax_zmx`).  Every
  example now runs to completion via the unit-test pinning suite.
* **B0-2 `_deprecation.py` shims wired** -- `load_zmx_prescription
  -> load_zemax_zmx` and `load_zemax_prescription_txt ->
  load_zemax_prescription_data_txt` aliases now emit a clear
  `DeprecationWarning` and forward to the new function.
  Pre-v4.12 these renamed functions raised cold
  `AttributeError`.

**B1 — Silently-wrong physics in default code paths**

* **B1-1 JAX/NumPy aperture schema unified** --
  `propagate_through_system_jax` now accepts the canonical NumPy
  schema (`diameter`, `width_x/y`, `inner_diameter/outer_diameter`,
  matching `apply_aperture`) AND the legacy JAX-only schema
  (`radius`, `half_width_x/y`, `inner_radius`) with a one-shot
  `DeprecationWarning`.
* **B1-2 `propagate_through_system_jax` fail-fast on
  non-traceable elements** -- up-front element-type scan raises
  `NotImplementedError` listing offending types
  (`spherical_lens`, `aspheric_lens`, `mirror`, etc.) BEFORE
  any tracing.  `_TRACEABLE_ELEMENT_TYPES = frozenset({'propagate',
  'lens', 'aperture', 'mask'})` exposed for programmatic checks.
* **B1-3 Rayleigh-Sommerfeld `z<=0` guard** -- matches existing
  Fresnel / Fraunhofer / SAS forward-only guards.  Pre-v4.12 RS
  silently produced 180-degrees-wrong-phase kernel for `z<0`.
* **B1-4 ASM-MFT band-limit `<=` -> `<` on NumPy** -- matches
  the JAX branch (and plain ASM).  Pre-v4.12 one-bin boundary
  disagreement between backends.
* **B1-5 SAS pad>2 centring** -- `as1 = (N_new - N) // 2`
  (was `(N+1)//2`, only correct for pad=2).  pad=4 now centres
  the input correctly.
* **B1-6 Dispatcher negative-z routing** -- `_auto_select_method`
  short-circuits to ASM for `z<0`; explicit
  `method='fresnel/fraunhofer/sas/rs'` with `z<0` raises a
  dispatcher-level `ValueError` naming `propagate`, not the
  underlying kernel.
* **B1-7 `propagate(return_result=True)` tuple unpacking** --
  `_coerce_field` rewritten to unpack `(E, dx_out, dy_out)` /
  `(E, dx_out)` tuples from Fresnel / Fraunhofer / SAS / Fresnel-
  MFT.  Result now reports the kernel's output dx, not the
  input dx.
* **B1-8 Dispatcher `output_grid` / `output_dx` for ASM family**
  -- auto-promotes ASM -> `angular_spectrum_propagate_mft`,
  Fresnel -> `fresnel_propagate_mft`, Fraunhofer ->
  `fraunhofer_propagate_mft`.  SAS / RS raise with guidance to
  use ASM-MFT.  Pre-v4.12 the kwargs were silently dropped.
* **B1-9 `_apply_doe_kick_jax` gradient flow** -- traced periods
  use `jnp.where(jnp.isfinite(period) & (period != 0), kick,
  0.0)` to keep gradient alive; concrete-period scalars stay on
  the Python branch.  `jax.grad` w.r.t. grating period now
  returns finite, non-zero gradients (within 1% of FD).
* **B1-11 `makedammann2d` global RNG** -- switched from
  `np.random.seed(seed)` to `np.random.default_rng(seed)` +
  `rng.random`.  No longer mutates the user's global RNG state.

**B2 — Silently-wrong physics in non-default paths**

* **B2-3 `image_plane_wfe` reference-sphere radius now includes
  `1/N_chief`** -- on-axis `N_chief=1` is a no-op; off-axis
  fields no longer get phantom-defocus absorbed by `best_rms`.
* **B2-4 `distortion_grid` raises on `sin(tx)^2 + sin(ty)^2 >= 1`**
  -- pre-v4.12 silently constructed N=0 rays, then swallowed
  the resulting trace failure via bare except.
* **B2-5 `apply_real_lens_traced` mirror guard** -- pre-flight
  scan raises with a properly-named error if any surface has
  `is_mirror=True` or `glass_after='MIRROR'`.  Points users at
  the per-segment `apply_mirror` pattern for folded designs.
* **B2-6 `gerchberg_saxton(backend='jax')` forwards
  `seed`/`dtype`/`initial_phase`** -- pre-v4.12 the dispatcher
  silently dropped these kwargs; function-level kwargs on
  `gerchberg_saxton_jax` were already wired internally.
* **B2-1/B2-2 `ghost.py` documentation** -- module + function
  docstrings now make explicit that `'intensity'` is an UPPER
  BOUND ignoring transmission losses
  (`I_true ~= I_reported * Prod (1-R_k)^2`), and
  `'focus_z_estimate'` is a heuristic harmonic-mean sort key,
  not a calibrated focal position.

### Known limitation deferred to v4.12.1

* **B1-10 Half-pixel grid convention drift** between propagator
  families.  ASM / Fresnel / RS / sources use pixel-centred
  `(arange(N) - N/2) * dx`; GBD / HF / subaperture / MHS /
  `optimize/core.py` use cell-centred `(arange(N) - N/2 + 0.5)
  * dx`.  Cross-method coherent superposition has a half-pixel
  shift producing wrong-physics phase error of order
  `k0 * dx/2 * off_axis_distance`.  Documented Tier-2 finding;
  per-site refactor scheduled for v4.12.1.

### Tooling

* New `benchmarks/` directory with `pytest-benchmark` per-area
  perf tests.  Run with `python -m pytest benchmarks/
  --benchmark-only -v`.  v4_11_2 baseline saved at
  `.benchmarks/v4_11_2_*.json`.
* Each optimization ships with a paired correctness-pinning
  test under `tests/unit/test_perf_v4_12_0_*.py`.
* Each round-4 audit fix ships with a paired pinning test under
  `tests/unit/test_audit_fixes_v4_12_0_round4_*.py`.
* `pytest-benchmark` added as a dev dependency.

### Discipline

Every fix / optimization gated by: (1) audit-claim verification
against actual code before fixing, (2) bit-near-exact (or LSB-
tolerant) correctness pin alongside the change, (3) targeted
validation suite pass before integration.  Two perf items that
didn't meet the cross-suite bar were reverted rather than
shipped.

## [4.11.2] — 2026-05-16

**Round-3 fresh-eyes audit response.**  An 11-agent fresh-eyes audit
of v4.11.1 (`AUDIT_ROUND3_2026_05_16.md`) surfaced ~120 new
substantive findings.  v4.11.2 closes ~70 of the highest-impact
findings across seven parallel work tracks plus a reconciliation
sweep, and ships ~55 new pinning regression tests.

The headline meta-finding is sobering: **the v4.10 "C-LR-1 fix" was
itself wrong**.  The pre-v4.10 sign on the Seidel-correction OPL
inside `apply_real_lens` was correct; the round-1 audit's
physics-reasoning step that justified flipping it was reversed.
v4.11.1's threshold tweak didn't help because the resulting bogus
correction was millimetre-scale, not nm-scale.  Reverted in this
release and pinned with a ground-truth regression test that compares
the analytic-screen path against `apply_real_lens_traced`.

### Critical-tier reversals and corrections

* **C-LR-1 reverted** (`_lens_real.py:867-895`): restored the pre-
  v4.10 negation `opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])`.
  The v4.10 patch had been producing a correction that approximately
  tripled the lens's analytic OPD at the rim.  Comment now spells
  out the physics under the library's `exp(-i*omega*t)` convention
  so this can't happen again.
* **GBD axial_opl** (`gbd.py:569-606`): v4.11.1's "dormant-fix
  activation" called `.get('thickness', ...)` on `Surface` dataclass
  instances (no `.get` method).  Switched to `getattr(_s, ...)`;
  the bare-except is narrowed and now emits a `RuntimeWarning`.
* **S-LAH64 / S-LAH79 Sellmeier coefficients** (`glass.py:172-182`):
  in-code coefficients gave `n_d` off by +5.8% and -5.9% vs the
  Ohara catalog.  Removed the misattributed entries; both glasses
  now route through the `__sellmeier__` sentinel + refractiveindex
  .info lookup.

### Critical-tier physics fixes

* **Chained-mirror Seidel parity** (`raytrace/core.py:2130-2192,
  3014-3128`): the v4.10 mirror-Seidel fix only handled a single
  mirror.  v4.11.2 tracks `mirror_parity = mirror_count % 2` through
  both `system_abcd` and `seidel_coefficients` so post-mirror glass
  indices carry the sign correctly.  Cassegrain / Schwarzschild /
  any 2-mirror catadioptric system now produces correct Seidel
  sums beyond the first mirror.
* **`system_abcd` ↔ `seidel_coefficients` mirror sign agreement**
  (`core.py:2130-2192`): both paths now use Welford's signed-R
  convention with `n2 = -n1`.  For R = -100 mm concave mirror,
  EFL = +50 mm (focusing) in both APIs.  Pre-v4.11.2 the two
  disagreed by sign.  `apply_mirror` retains its own (magnitude-
  based) `R > 0 = concave` convention -- documented in its
  docstring; the user-facing wave-side API stays unchanged.
* **`seidel_wfe` field-curvature DC term added** (`seidel_analysis.py:
  158-187, 290-300`): the Hopkins/Welford expansion has BOTH
  `(1/2)·S₃·ρ²·cos²θ` (astigmatism, already present) AND
  `(1/4)·S₃·ρ²` (field-curvature DC, missing).  Both docstring and
  implementation updated.
* **EVENASPH PARM off-by-one in Zemax loader** (`prescriptions.py:
  578-592`): pre-v4.11.2 the filter dropped `PARM 1` (the dominant
  α₄) entirely and mis-labelled higher coefficients.  Every Zemax-
  authored EVENASPH file ever loaded by Lumenairy had silently lost
  its α₄.  The exporter at line 1506 was already correct (so a
  round-trip via Lumenairy lost α₄ on load and then put zero in
  the export); now both sides agree on `power = 2 + 2*parm_num` with
  filter `>= 1`.  Tested with round-trip pin.
* **Quadoa aspheric serializer** (`prescriptions.py:2130-2172`):
  iterated `coeffs` (a dict) yielding keys not values, so the JSON
  carried `[4.0, 6.0, 8.0]` (the powers) instead of the coefficient
  amplitudes.  Rewrote to emit `{str(power): value}` dict + paired
  deserializer; aspheric_coeffs now harmonised to dict form across
  all loaders.
* **`normalize_prescription` mirror filter** (`prescriptions.py:
  2579-2584`): checked `e.get('mirror')` but the library uses
  `element_type='mirror'`.  Filter was a no-op; mirror entries
  flowed through to `apply_real_lens` un-flagged.  Fixed.
* **Zemax exporter STOP marker on folded designs** (export path):
  pre-v4.11.2 the STOP-index counter included coord-breaks and
  mirrors; `stop_surface=N` (zero-based among refractives) landed
  on the wrong surface for any folded design.  Now uses a separate
  refractive-only counter.
* **Mirror / coord-break DISZ round-trip** (export path): pre-v4.11.2
  applied a `mirror_count`-parity flip that double-negated mirror
  and post-mirror coord-break thicknesses.  Removed; canonical
  thicknesses are Zemax-signed as of GUI v3.7.4 already.
* **`propagate_hfpi_through_prescription` finite-conjugate dead
  path** (`hfpi.py:690-692`): paths were initialised at z=0 then
  back-propagated to `z=-object_distance`; the `t >= 0` mask killed
  them all.  Function only "worked" for `object_distance=0`.  Now
  initialises directly at `z_input_plane = -object_distance`.
* **`init_paths_stratified` cartesian product** (`hfpi.py:493-505`):
  the `np.repeat` pattern only enumerated `(0,0,0,0)` and
  `(1,1,1,1)` of the 16 strata.  Replaced with
  `np.indices(...).reshape(4, -1)`.
* **Richards–Wolf prefactor** (`vector_diffraction.py:221`):
  prefactor was `(-1j * k * f / (2π)) · exp(-ikf)` -- both the
  multiplicative `f` (instead of `1/f`) and the `exp(-ikf)` sign
  were wrong.  Fixed to `(-1j * k / (2π * f)) · exp(+ikf)`.
  Intensity now scales as `1/f²` and the global phase matches the
  rest of the library's forward propagators.
* **`compute_psf` Parseval default broke `t_strehl_perfect`**:
  test was asserting `psf.max() > 0.99` but with the `'power'`
  default the peak was ~89795 (so the assertion was passing for
  the wrong reason).  Kept `'power'` default (documented v3.1.1+
  semantic); updated `t_strehl_perfect` to request `normalize='peak'`
  explicitly and tightened the assertion to `abs(peak − 1) < 1e-9`.

### High-tier dead-on-arrival / sibling-function omissions

* **`propagate_huygens_fresnel_with_opl_callable`** missing `-1j`
  Maslov prefactor (sibling of `propagate_hf_chebyshev_quadrature`).
* **`propagate_huygens_fresnel_through_prescription(method=
  'asymptotic')`** silently replaced `E_in` with a fundamental
  Gaussian.  Now decomposes `E_in` to LG modes via `decompose_lg`
  with kwargs `source_lg_p_max`, `source_lg_ell_max`,
  `source_lg_amp_threshold`.
* **HFPI Kirchhoff `1/(iλ)·dΩ`** added to `apply_aperture_diffraction`,
  `init_vector_paths_from_field`, `apply_vector_aperture_diffraction`
  (init_paths_from_field already had it from v4.10).
* **HFPI RNG per-aperture** via `np.random.SeedSequence` /
  `Generator.spawn` / `jax.random.fold_in`.  Pre-v4.11.2 the master
  seed was re-used at every aperture, producing perfectly correlated
  draws across diffraction events.
* **Asymptotic Maslov branch-tracking** hoisted to a shared helper
  (`_maslov_branch_corrected_sqrt`); `aberration_tensor` (NumPy)
  routes through it.  JAX twins documented as principal-sqrt with
  parity at the single-shot point.
* **`maslov_tracking` kwarg** on `propagate_modal_asymptotic` with
  `{'principal', '1d_raster', 'row_reset'}`; default `'row_reset'`
  resets the unwrap state at every row to avoid the 2-D raster's
  spurious row-wrap flips.
* **Subaperture per-patch fit** now passes `source_centre=(cx_i, cy_i)`
  so the polynomial is trained around each patch's centroid, not
  globally (was producing zero-field outside the global axial box).
* **Real-`E_in` dtype promote** in HF Chebyshev quadrature so the
  kernel imaginary part isn't stripped.
* **AO rim Zernike FD** extended to all four rim quadrants (was
  +x/+y only).
* **EP-aiming siblings**: ported the v4.10 H-AB-3 fix to
  `relative_illumination`, `field_aberration_sweep`, `ray_fan_data`,
  `opd_fan_data`.  Pre-v4.11.2 chief was launched at (0,0,0); now
  aimed at the EP centre derived from `first_order_data` with a
  legacy fall-back if pupils can't be computed.
* **Phase-retrieval seed= parity**: `gerchberg_saxton_jax` actually
  consumes `seed` via `jax.random.PRNGKey`; NumPy `error_reduction`
  and `hybrid_input_output` accept `seed=` / `dtype=` and enforce
  them.
* **`apply_real_lens_traced` M_x / M_y indices** transposed
  (`_lens_traced.py:1789-1792`).
* **Freeform terms** in thin-element `apply_real_lens` emit a
  `RuntimeWarning` instead of silently dropping (the ray-traced /
  Maslov paths already honoured freeform through the Surface
  dataclass's `freeform` field).
* **`stop_index != 0`** in `apply_real_lens_traced` and
  `apply_real_lens_maslov` warn instead of silently ignoring.
* **NaN sentinel leak** from aspheric clamp masked to 0 before
  `exp(-i·k·NaN)`.
* **Coating `'avg'` mode** stores `eta_s`, `eta_p` separately per
  polarization.
* **`apply_waveplate` docstring** synced to match the actual
  implementation `R(θ)·diag(1,exp(-iφ))·R(-θ)`.
* **`spot_diagram` / `trace_summary` Airy radius** now includes
  `|f_eff|` factor; pre-v4.11.2 was a half-angle in radians
  mis-labelled as a length.
* **`bundles.py` conversion helpers** rewritten to use the actual
  `RayBundle` attributes (`x, y, z, L, M, N`); pre-v4.11.2 called
  `.positions` / `.directions` which don't exist.
* **`find_best_focus` NaN guard**, `monte_carlo_tolerancing_linearized`
  clamp `a_k >= 0` (Marechal invariant), `compute_psf` non-square
  pupil clear error, `apply_detector` non-integer pixel ratio area
  scale correction, `polychromatic_strehl` / `polychromatic_psf`
  honour `get_default_complex_dtype()`.
* **codegen** `op.GLASS_REGISTRY` → `la.GLASS_REGISTRY`; system-list
  style now emits `aperture_diameter` for mirrors.
* **`load_material`** emits a `RuntimeWarning` when a saved
  dispersion field is being dropped on load.
* **`Source.*` classmethods** propagate `**factory_kwargs` to the
  underlying create_* factories so `dy=`, `dtype=`, `normalize=`,
  etc. work.

### Test-suite + harness improvements

* `tests/unit/test_audit_fixes_v4_11_2_*.py` — ~55 new pinning tests
  across raytrace, IO, RW+lens, HFPI+HF+asymptotic, analysis,
  and Track A (C-LR-1 / GBD / S-LAH).
* Strengthened three v4.11.1 pinning tests that passed for the wrong
  reason: MultiWavelengthMerit chromatic semantics (was warning-
  absence only), Subaperture actual call (was import-only),
  Tilted-ASM bandlimit (tilt too small to trigger pre-fix path).
* New `RS-vs-ASM phase pinning test` at z > 0 with `bandlimit=True`
  -- the single biggest test-coverage gap per the round-3 audit.
* `validation/_harness.py:33` `warnings.simplefilter('ignore')`
  scoped to `DeprecationWarning`, `PendingDeprecationWarning`,
  `ResourceWarning`, `ImportWarning` only -- `RuntimeWarning`,
  `UserWarning`, numerical/overflow warnings now propagate.
* `validation/.../t_dammann_grating`, `t_prescription_hf_asymptotic`,
  `t_subaperture_asymptotic_singlet` had bare `except: return True,
  'skipped'` patterns that hid genuine failures.  Removed; tests
  now propagate real exceptions and the Dammann test was rewritten
  to current API.

### Test results

* All 243 unit tests pass.
* Full `validation/run_all.py` (34 files, 314 tests).
* Pre-existing failure in `validation/elements/test_elements.py`'s
  "Mirror: f = R/2 for concave mirror" test resolved by switching
  the test to Welford signed-R convention (the test was pinning the
  legacy bug); the test now correctly asserts `EFL = +|R|/2` for a
  signed-R = -100 mm concave mirror.

### Documented limitations (carried over from v4.11.1)

* `_transfer_jax` paraxial form retained (math-correct form
  NaN-poisons `jax.grad` through `fit_canonical_polynomials_jax`).
* `aberration_tensor` axial-multi-p ℓ=0 saddle-point degeneracy
  emits a `RuntimeWarning`.

## [4.11.1] — 2026-05-16

**Round-2 verification follow-up: close the residuals from
`AUDIT_VERIFICATION_2026_05_16.md`.**  The round-2 verification of
the v4.10 / v4.11.0 fix wave identified five fixes that had landed
dead-on-arrival (call-signature mistakes or wrong-API lookups), four
new bugs the fix wave introduced, three unfixed silent-failure
paths in the JAX trace, and one over-coarse threshold from the
4.10 Seidel-correction sign fix.  4.11.1 closes all of these and
ships the first round of pinning regression tests (zero new test
files in the entire v4.10 series).  All 179 unit tests pass.

### Dead-on-arrival fixes (4.10 series)

* **C-OP-1 / N1**: `MultiWavelengthMerit` called the per-wavelength
  `apply_real_lens(E, ctx.prescription, wl, dx_pix)` positionally,
  but `apply_real_lens` is keyword-only after `E_in` since 4.7.
  Every iteration raised `TypeError`, swallowed by a bare
  `except Exception: pass`, so the per-wavelength wave-leg silently
  reused the parent's single-wavelength values.  Chromatic
  optimisation was a no-op for the entire v4.10 series.  4.11.1
  passes by keyword and narrows the except to typed warnings so
  the failure mode is visible if it ever recurs.
  (`lumenairy/optimize/core.py:1927`)
* **M-LR-1**: decentered-stop fix at `_lens_real.py:691-692` called
  `getattr(surf, 'decenter_x_m', 0.0)` on a *dict*.  `getattr` on a
  dict for a non-attribute name silently returns the default, so
  the stop stayed on-axis.  4.11.1 reads `surf.get('decenter')`
  (the actual key, value is a `(dx, dy)` tuple).
* **C-PL-1**: the 4.10 swap of `create_circular_polarized` flipped
  `'right'` to `(1, -i)/sqrt(2)`, which under the library's
  `S3 = -2 Im(Ex Ey*)` convention gives `S3 = -1` -- contradicting
  the docstring ("S3 > 0 for right"), `apply_waveplate(QWP, 45°)`
  on a linear-x input (which produces `(1, +i)/sqrt(2)`), and
  `vector_diffraction.py:147`'s hard-coded right-circular Jones
  vector.  4.11.1 restores the consistent `(1, +i)/sqrt(2)` form
  for `'right'` and updates the docstring.

### New bugs introduced in v4.10

* **N2**: Richards-Wolf rim mask `sin_theta <= sin(theta_max)` was
  built *after* `sin_theta` was clipped to `sin(theta_max)`, so the
  mask was identically `True` over the whole grid and the geometric
  pupil went unenforced.  4.11.1 builds the mask from the unclipped
  `sin_theta_raw = rho_p / f` before any clipping.
  (`lumenairy/propagators/vector_diffraction.py:118,137`)
* **N3**: `_sag_derivatives_param` lacked `sign(R)` (the C-RT-3
  fix was applied only to the static `_sag_derivatives_jax` twin),
  so concave conic / aspheric surfaces traced through the
  differentiable `trace_jax_with_params` /
  `fit_canonical_polynomials_jax` got the wrong transverse-normal
  sign.  4.11.1 mirrors the fix using `jnp.where(R >= 0, 1, -1)`.
* **N4**: `_intersect_jax_param` Newton step used the single-where
  pattern `jnp.where(|dF_dt|>eps, F/dF_dt, 0.0)`, which still
  evaluates the division on the False branch and NaN-poisons
  `jax.grad` when `dF_dt → 0` at grazing rays.  4.11.1 uses the
  double-where idiom (substitute `dF_dt = 1` on the stuck branch
  *before* division), mirroring the static `_intersect_jax`.
* **N5**: `subaperture.py:281-285` built `output_grid_xy =
  np.stack([OX, OY], axis=-1)` (`ndim=3`) and then tried to unpack
  it `sgx, sgy = output_grid_xy`, which raised `ValueError` for
  any `Ny != 2` grid.  The subaperture-asymptotic path was dead
  on call.  4.11.1 simplifies to `sgx, sgy = OX, OY`.
* **N6**: `np.argmax(scan.strehl)` in `MultiWavelengthMerit` and
  `MultiFieldMerit` (`optimize/core.py:1940, :2034`) was sensitive
  to NaN slices.  4.11.1 uses `np.nanargmax` guarded by an explicit
  `np.any(np.isfinite(...))` check.
* **N7**: `MultiWavelengthMerit` and `MultiFieldMerit` hard-coded
  `np.complex128` (`:1926, :2013`), silently negating the
  `precision='single'` knob.  4.11.1 honours `get_default_complex_dtype()`.
* **N8**: `aperture_diameter` fallback used `or` instead of an
  explicit `is None` check (`:1924`), so an aperture set to `0.0`
  (a legitimate sentinel) was silently overridden with the
  grid-arbitrary default.
* **N9**: bare `except Exception: pass` around the
  `MultiWavelengthMerit` wave-leg block (`:1916, :1945-1946`)
  hid C-OP-1 for the entire v4.10 series.  4.11.1 narrows to
  `(TypeError, ValueError, RuntimeError)` and emits a
  `RuntimeWarning` with the wavelength and exception text on
  fallback.

### Still-unfixed-despite-release-notes paths

* **H-RT-5**: `_intersect_jax` had no `~isfinite(t)` mask, no
  `disc < 0 → alive=False`, and no Newton-stuck → `alive=False`,
  despite the 4.10 release-notes claim.  4.11.1 tracks a `miss`
  flag through every branch and propagates it into the returned
  `state.alive`.  Mirrored in `_intersect_jax_param`.
* **H-RT-7**: `sqrt(maximum(disc, 0))` at `jax_trace.py:205` (and
  `:691` in the param twin) has gradient `1/(2 sqrt(0)) → ∞` at
  the disc=0 tangent-ray boundary, NaN-poisoning `jax.grad` for any
  ray that grazes a sphere.  4.11.1 substitutes the double-where
  idiom on both sites.
* **M-RT-3**: `_refract` / `_reflect` stamped `RAY_NAN` into
  `error_code` on direction-vector collapse but left `rays.alive
  = True`, so the dead ray continued through subsequent surfaces
  with its last-valid direction.  4.11.1 also clears `rays.alive`
  on the degenerate mask.
* **H-PR-4**: `create_point_source` clamped the singular `r` at
  `1e-30`, producing `|E_central| = amplitude / 1e-30 ≈ 1e30` for
  `|z0| < dx`.  4.11.1 clamps `r` at the pixel half-diagonal
  `sqrt(dx²+dy²)/2`, capping `|E_central|` to `~ amplitude / dx`
  (the physically correct discretisation-aware scale).

### Dormant fix activated

* **H-AS-1**: `apply_abcd_to_beamlets` has accepted `axial_opl=`
  since 4.10.2, but `propagate_gbd_through_prescription` never
  populated it.  4.11.1 computes
  `axial_opl = sum_k n_k * thickness_k` over each segment using
  `surfaces_from_prescription` + `get_glass_index` and passes it
  through, so the reconstructed field carries the system's
  absolute axial phase reference (matters for cross-method
  comparisons with ASM / Fresnel).

### Threshold / docstring follow-ups

* **C-LR-1 follow-up**: the 50 nm Seidel-correction RMS gate in
  `apply_real_lens` was set when the 4.10 sign-flip bug routinely
  produced corrections of O(λ).  After the sign fix, real
  residuals collapsed to a few-nm range and 50 nm silently skipped
  every meaningful correction.  4.11.1 drops the gate to 5 nm.
* **RS docstring**: the kernel formula at `propagation.py:2663`
  still showed the pre-4.10 `(ik - 1/r)` form; the code uses the
  correct Goodman 3-43 `(1/r - ik)`.  4.11.1 updates the docstring
  to match.
* **`seidel_wfe` docstring**: the displayed formula used
  `σ²·ρ²`; the code has always used `H²·ρ²` (Lagrange invariant),
  which equals `σ²·f_eff²` in the small-angle limit but is the
  right invariant for finite-conjugate and stop-shifted systems.
  Docstring corrected with a one-paragraph note on the relationship.
* **`_transfer_jax` accuracy bound**: the documented "~1% for
  NA ≤ 0.1" was a per-surface estimate; over a 5-surface trace the
  error accumulates to ~2.5%.  Docstring updated to "~0.5% per
  surface, accumulates" with the explicit
  `~ thickness * NA² / 2` per-surface scaling.

### Regression-test coverage

* `tests/unit/test_audit_fixes_v4_11_1.py` adds 9 pinning tests:
  - `MultiWavelengthMerit` does not silently fall back to parent
    wave-leg values (positive test for C-OP-1 / N1).
  - Decentered stop is honoured in `apply_real_lens` (clips at
    the offset disk, not the optical axis).
  - `create_circular_polarized('right')` has `S3 > 0`.
  - `apply_quarter_wave_plate` on linear-x at fast-axis π/4
    matches `create_circular_polarized('right')` handedness.
  - `create_point_source` central pixel `|E| < 1e7` when
    `|z0| < dx`.
  - `propagate_subaperture_asymptotic` is importable.
  - Concave mirror has non-zero `S4` (Petzval).
  - Tilted-ASM bandlimit yields non-zero rms output.
  - `trace_jax` raises `NotImplementedError` on a mirror surface.

These were specifically called out as the "no test coverage" gap
in the round-2 verification.

## [4.11.0] — 2026-05-16

**Roll-up release for the v4.10 audit-response series.**  Five
patch releases (4.10.0, 4.10.1, 4.10.2, plus the in-tree work
toward 4.10.3) addressed ~100 audit findings from three converged
audit sources.  4.11.0 ships the final three deferred items plus
the cumulative work as a minor version bump for PyPI / GitHub
release tagging.  All 34 validation files (314 tests) pass.

### Final wave (4.10.3 / 4.11.0 work)

* **C-AS-1 partial fix**: ``aberration_tensor`` closed-form ℓ=0
  path now evaluates the output LG polynomial at the saddle's
  ``σ_image`` instead of grabbing only its ``(0,0)`` Cartesian
  constant.  Different ``(p, 0)`` modes are distinguished for
  any OFF-axis saddle.  The ON-axis multi-p case emits a clear
  ``RuntimeWarning`` -- this is a fundamental saddle-point limit
  (LG_p,0 modes all peak at the origin), not a code bug.
* **H-PR-2**: ``through_focus_scan_jax`` replaces its Python
  for-loop over z with a proper ``jax.vmap`` over an inline ASM
  kernel.  Speedup is ~5-15× for typical 30-point scans, larger
  on GPU.  Output values are bit-identical to the loop version.
* **C-RT-2 (deferred again, documented)**: ``_transfer_jax``'s
  math-correct ``t = (thickness − z) / N`` form still
  NaN-poisons ``jax.grad`` through
  ``fit_canonical_polynomials_jax`` even with triple-where
  guards and ``jnp.isfinite`` filtering on ``t``.  Multiple
  investigation attempts (4.10.0 → 4.10.3) have not isolated
  the gradient-graph issue to a specific op; the paraxial form
  ``x += L·thickness`` is retained.  For NA ≤ 0.1 (the typical
  LumenAiry use case) the two forms agree to ~1 %.

### Cumulative summary (4.10.0 → 4.11.0)

This minor release tags the entire v4.10 audit-response series.
Earlier waves' details remain in the per-release sections below.
Headline impact:

* **Critical / silent-wrong-physics bugs fixed**: mirror Seidel
  zeros, exit-pupil radius inversion, RS sign flip, tilted-ASM
  bandlimit miscentring, Richards-Wolf Jacobian + prefactor,
  coord-break order, Lagrange invariant for finite-conjugate
  systems, MultiWavelengthMerit chromatic no-op, mutual_coherence
  conjugate flip, Shack-Hartmann wavefront units, circular-
  polarization handedness, sagittal/tangential fan swap, GBD
  tilt-phase ramp, HFPI Kirchhoff weighting, compute_psf Parseval,
  TIS Monte-Carlo cos factor, ghost interferometry fringe formula,
  register_fixed_glass without refractiveindex, MC tolerancing
  quadratic Marechal prediction.
* **High-tier silent failures fixed**: 30+ items including
  apply_axicon import, aspheric clamp NaN, JAX trace surface-
  type guards, OPL chief-ray pick, image-plane WFE aim-at-EP,
  through-focus JAX/NumPy parity, JAX error_reduction ordering,
  thin-film transmission for absorbing stacks, Snell complex-n,
  precision='single' end-to-end through merits, LM residual
  differentiable at zero, decentered-stop h_sq.
* **Medium / Low items addressed**: OSA Zernike doc, dy support
  on sources, NA-aware sampling check, plotting per-axis padding
  + log floor, Sellmeier resonance validation, point-source clamp
  warning, fiber-NA warning, LED-source sample-count doc,
  caustic-thickness length assertion, Maslov-branch tracking for
  caustic-continuous phase, ASM-MFT bandlimit ``<`` consistency,
  user_library eval safety, plot dx/dy axis labels, mutual-
  coherence conjugate.

### Items NOT addressed (and why)

* **C-AS-1** axial multi-p ℓ=0: saddle-point fundamental limit.
  Warning emitted; off-axis case fully fixed.
* **C-RT-2** JAX ``_transfer_jax`` math-correct form: gradient
  instability whose root cause needs deeper investigation.
* **H-SC-2** SAS odd-N padding: only affects rare odd-N grids.
* **H-GL-3** ``least_squares`` ``method='lm'`` → ``'trf'`` switch
  with bounds: scipy's documented behaviour.

See individual release notes below for per-wave details and
file:line citations.

## [4.10.2] — 2026-05-16

**Wave 5 of the v4.10 audit response.**  Closes the remaining
Critical / High / Medium audit items that the first four waves left
unaddressed (the user asked to bring "all changes from the converged
audits" in; 4.10.2 covers the residuals).  All 34 validation files
(314 tests) pass.

### Critical residuals

* **``register_fixed_glass`` works without refractiveindex**
  (``glass.py``).  Pre-4.10.2 the user-registered ``('__user__',
  '__fixed__', '__fixed__')`` sentinel fell through to the
  refractiveindex.info dispatch branch and raised ``ImportError`` on
  minimal installs.  The dispatch now recognises the sentinel and
  returns from ``_glass_cache`` directly.
* **Monte-Carlo tolerancing uses quadratic Marechal prediction**
  (``analysis/through_focus.py``).  ``monte_carlo_tolerancing_linearized``
  now fits a per-knob ``a_k = (S_nom − S(σ)) / σ²`` coefficient and
  superposes ``S_pred = S_nom − Σ a_k · ξ_k²`` (Marechal:
  ``S ≈ exp(−σ_φ²)``).  Pre-4.10.2 used a linear FD + linear
  superposition that produced a mean-zero distribution around
  ``S_nom`` -- the wrong physics.

### High residuals

* **GBD axial OPL phase** (``propagators/gbd.py``).
  ``apply_abcd_to_beamlets`` accepts an ``axial_opl=`` kwarg that
  injects ``exp(+i·k·L_chief)`` into ``qratio``.  ``propagate_gbd_
  through_prescription`` should pass the system OPL for absolute-
  phase reconstruction; pre-4.10.2 the phase was a constant piston
  bug only visible against an external reference arm.
* **GBD thin-lens slope/direction-cosine** (``propagators/gbd.py``).
  ``apply_thin_lens_to_beamlets`` now does the lens kick on
  paraxial slopes ``u = L/N`` and re-normalises to direction
  cosines, not directly on direction cosines.  For NA ~ 0.05-0.1
  this is a few-percent correction that compounds across
  surfaces.
* **HFPI symmetric obliquity** (``propagators/hfpi.py``).
  ``apply_aperture_diffraction`` weights the secondary HF sources
  by ``(cos θ_in + cos θ_out)/2``, the symmetric Kirchhoff form.
  Pre-4.10.2 used only ``cos θ_out`` relative to +z -- correct for
  a single normal-incidence aperture, wrong for cascaded apertures
  with oblique paths.
* **``apply_jones_matrix`` shape guard** (``elements/polarization.py``).
  Now raises ``ValueError`` when a callable returns a non-
  ``(2, 2, Ny, Nx)`` shape.  Permits the swapped
  ``(Ny, Nx, 2, 2)`` layout via auto-transpose.  Pre-4.10.2
  silently broadcast any shape and produced wrong answers.
* **``SphericalSeidelMerit`` NaN guard** (``optimize/core.py``).
  Reads ``ctx.seidel`` through ``np.isfinite`` and returns the
  default weight (instead of NaN) when the upstream Seidel
  computation produced the 4.10.1 NaN sentinel.  Prevents scipy
  from refusing the objective.
* **``ChromaticFocalShiftMerit`` decoupled** (``optimize/core.py``).
  New ``wavelengths=`` constructor kwarg makes the term self-
  contained -- per-wavelength EFL is computed from the
  prescription via ``system_abcd``.  Pre-4.10.2 the term depended
  on ``ctx.efls_per_wavelength`` being populated as a side effect
  of a prior ``MultiWavelengthMerit.evaluate()`` call; ordering
  the terms differently silently disabled the constraint.
* **``precision='single'`` actually halves precision through
  merits** (``optimize/core.py``).  ``MatchIdealSystemMerit`` and the
  tolerancing inner loop allocate ``E_in`` at ``get_default_complex_
  dtype()`` rather than the hard-coded ``np.complex128``.
* **LM residual differentiable** (``optimize/core.py``).
  ``np.sqrt(max(m.evaluate(ctx), 0.0))`` now uses a tiny ``1e-30``
  floor before the sqrt so the residual is differentiable
  everywhere; FD Jacobian no longer produces inf/nan columns
  near a converged solution.

### Medium residuals

* **Decentered stop respected by ``apply_real_lens``**
  (``elements/_lens_real.py``).  Stop aperture mask now uses
  ``(x − xc_stop)² + (y − yc_stop)²`` when the stop surface has a
  non-zero ``decenter_x_m`` / ``decenter_y_m``.  Pre-4.10.2 always
  used the axis-centred ``h_sq_axis``, clipping the wrong region.

### Items NOT addressed (with documented rationale)

* **C-AS-1** (asymptotic closed-form ℓ=0 path): 4.10.0 added a
  warning; the σ-grid would be the correct fix but breaks the
  JAX-backend hard-coded twin's numerical parity with NumPy.
  Tracked as a known limitation; pass at least one ``ℓ ≠ 0`` mode
  to force the correct path manually.
* **C-RT-2** (JAX ``_transfer_jax`` math-correct form): 4.10.0
  documented as a known limitation -- the math-correct
  ``t = (thickness − z) / N`` form introduces a gradient
  instability through ``fit_canonical_polynomials_jax`` whose root
  cause needs deeper investigation.  Paraxial form retained,
  accurate to ~1 % for NA ≤ 0.1.
* **H-SC-2** (SAS asymmetric padding for odd N): only affects
  odd-N grids which are rare in LumenAiry workflows; cosmetic
  half-pixel shift fixed by using even-N grids.
* **H-PR-2** (``through_focus_scan_jax`` "vmap" is a Python loop):
  performance issue, not correctness.  Pre-existing acknowledged
  TODO.
* **H-GL-3** (``least_squares`` ``method='lm'`` silently switches to
  ``'trf'`` with bounds): documented behaviour of scipy itself;
  4.10.2 leaves it as a non-issue.

## [4.10.1] — 2026-05-16

**Wave 4 of the v4.10 audit response.**  After the first three waves
landed, the user requested coverage of every High/Medium-tier finding
that hadn't been addressed yet.  4.10.1 closes the remaining residual
audit items.  All 34 validation files (314 tests) still pass; the
fixes here primarily affect non-default code paths and edge cases.

### Tier 1 fixes (silent wrong physics in feature-complete code)

* **HFPI** (``propagators/hfpi.py``): Kirchhoff prefactor ``1/(i*lambda)``
  and solid-angle Monte-Carlo weight ``2*pi*(1-cos(theta_max))/N_paths``
  are now applied to each emitted path.  Absolute amplitudes were
  unphysical by ~10⁶ per re-emission at visible wavelengths pre-4.10.1
  (relative-contrast results within a single experiment were
  unaffected; cross-experiment / absolute-photometry use is new).
* **MHS ASM subdomain** (``propagators/mhs.py``): ``asm_subdomain`` now
  raises ``ValueError`` when ``in_surface.dx != out_surface.dx``.  ASM
  preserves pitch; pre-4.10.1 silently labelled the output field with
  the wrong dx, corrupting any downstream subdomain that consumed the
  labelled coordinates.
* **Detector pixel integration** (``analysis/detector.py``): replaced
  ``scipy.ndimage.zoom(order=1)`` (point-sample bilinear interpolation,
  NOT area-preserving) with proper area integration -- block-sum
  ``np.add.reduceat`` for integer pixel ratios, uniform-filter +
  centred sample for non-integer ratios.  Photon conservation now
  holds for arbitrary ``pixel_pitch/dx_field`` ratios.
* **Image-plane WFE aim point** (``analysis/image_plane_wfe.py``): rays
  now aim at the entrance pupil ``(px*ep_radius, py*ep_radius, ep_z)``
  instead of ``(px*semi, py*semi, 0)``.  For stop-at-front systems the
  two coincide; for mid-stop systems pre-4.10.1 landed off-axis rays
  at the wrong pupil position and reported wrong WFE.  Also: chief-ray
  pick now skips dead rays (was occasionally NaN-poisoning the full
  result), and the best-RMS sphere shift uses the actual sphere radius
  (``img_d_m - fod.xp_z`` for ``sphere_tangent='exit_pupil'``) instead
  of ``img_d_m`` directly.
* **``compute_psf`` Parseval correction** (``analysis/core.py``):
  ``normalize='power'`` now enforces physical Parseval
  ``sum(|E_pupil|^2)*dx_pupil^2 == sum(|E_psf|^2)*dx_psf^2``.  Pre-4.10.1
  enforced equal pixel-sums, off by ``(dx_pupil/dx_psf)^2``.  Strehl
  ratios cancel the constant; absolute-photon-flux consumers now
  match the documented contract.

### Tier 2 fixes (silent inconsistencies)

* **Shack-Hartmann calibration** (``analysis/detector.py``): adds a
  reference-centroid pass on a flat-wavefront field and subtracts the
  per-lenslet bias from every measurement.  Out-of-bounds lenslets are
  now NaN-flagged (not zero-filled).  Cumulative-integration step
  zero-pads NaN slopes so OOB lenslets don't NaN-poison the wavefront.
* **FD gradient step scaling** (``optimize/core.py``): central
  differences (vs forward) and per-variable scale floors pulled from
  ``parameterization.scale_floor`` (default 1 micron for radii /
  thicknesses).  Pre-4.10.1 the ``max(|x|, 1.0)`` floor pinned all
  steps at 1e-7 regardless of variable type, biasing L-BFGS-B's
  Hessian estimate.
* **``MinThicknessMerit`` air-gap exclusion**: by default only GLASS
  thicknesses are penalised; air gaps that legitimately need to be
  small (cemented interfaces) no longer trip the constraint.
  ``include_air=True`` restores the pre-4.10.1 behaviour.
* **``apply_real_lens`` cos-clamp warning**: 1e-3 ≈ 89.94° AOI floor
  now emits a one-time ``RuntimeWarning`` when triggered on real
  (non-TIR) rays, so kilo-radian-of-phase clipping no longer hides
  silently.
* **Phase-retrieval seed + dtype** (``analysis/phase_retrieval.py``):
  NumPy and JAX variants accept ``seed=`` and ``dtype=`` so backends
  are interchangeable and runs are reproducible.  Default dtype on
  the JAX path remains float32 for back-compat; pass
  ``dtype=np.float64`` for NumPy parity.
* **AO docstring example** (``analysis/ao.py``): the module-doc
  example now correctly unpacks ``shack_hartmann``'s 5-tuple return.
* **Caustic-diagnostic length assertion** (``analysis/aberration.py``):
  rejects mis-matched ``surfaces`` / ``thicknesses`` instead of
  silently mis-placing sample planes via the cumulative-z table.
* **NA-aware sampling check** (``analysis/core.py``):
  ``check_sampling_conditions`` accepts ``NA=`` and relaxes the
  Nyquist criterion to ``dx < lambda/(2*NA)`` (was hard-coded ``NA=1``,
  flagging valid setups as under-sampled).

### Tier 3 fixes (cosmetics / edge cases)

* **Asymptotic** (``propagators/asymptotic.py``): Maslov-branch
  tracking on ``sqrt(det M)`` across pixels so the reconstructed
  field stays phase-continuous through caustics.  Newton stall check
  rewritten as an explicit two-condition expression instead of the
  buggy chained comparison.  Removed unused ``Sigma = 0.5 * M_inv``.
* **Plain-ASM cache key** (``propagators/propagation.py``): added
  ``'ASM'`` tag for parallelism with the other propagators' keys.
  ASM-MFT bandlimit now uses strict ``<`` (Matsushima open-interval),
  matching plain-ASM.
* **Plotting** (``analysis/plotting.py``): ``auto_crop`` pads each
  axis from its own extent (was using row extent for both axes).
  ``plot_cross_section(log=True)`` guards against all-zero / NaN
  intensity inputs.
* **Source warnings** (``sources/core.py``): ``create_point_source``
  warns when ``|z0| < dx`` (where the central-pixel clamp dominates
  integrated power); ``create_fiber_mode`` warns when ``NA > 0.2``
  (Gaussian-MFD approximation breaks down for LP01); LED source
  docstring corrected to report 37 angle samples (not "~21").
* **Glass** (``glass.py``): missing-extinction-coefficient catalogue
  entries now emit a one-time per-glass ``RuntimeWarning`` instead
  of silently falling back to ``kappa = 0``.

### Known limitations

* ``_transfer_jax`` retains the paraxial-approximate form ``new_x =
  x + L*thickness`` (vs the math-correct ``t = (thickness - z)/N``).
  The math-correct form introduces a gradient instability through
  ``fit_canonical_polynomials_jax`` whose root cause is still being
  investigated; the paraxial form is accurate to ~1 % for NA ~ 0.1.
* ``aberration_tensor`` closed-form ``ℓ=0`` path emits a warning when
  multiple radial orders ``p`` are in the output-mode list, but does
  not auto-route to the σ-grid path (which would break JAX-backend
  parity with the current closed-form).  Pass at least one ``ℓ ≠ 0``
  mode to force the correct path manually.

## [4.10.0] — 2026-05-16

**Multi-agent physics audit response.**  4.10 closes ~50 audit
findings drawn from three independent audit runs (one external,
two internal multi-agent) of the 4.9.0 codebase.  Severities ranged
from "silent wrong answer in a default code path" through "wrong
units" to documentation drift.  Every fix is gated by the 34-file
validation suite (314 tests passing, no regressions).  The full
unified plan lives in ``CORRECTION_PLAN.md``.

### Critical & high-impact fixes

* **Mirror Seidel S1..S5 now populated** (``raytrace/core.py``).  The
  mirror branch in ``seidel_coefficients`` previously updated only
  ray heights, leaving S1..S5[i] = 0 for every reflective surface.
  Every catadioptric / reflective system silently reported "well-
  corrected"; that's now fixed using the Welford form with n2 = -n1.
* **Exit-pupil radius**: was ``stop_radius * D_post`` (the angular
  magnification), should be ``stop_radius * det(M) / D_post`` (the
  transverse magnification, = 1/D for det = 1).  Every downstream
  Seidel / vignetting / f/# consumer was off by 1/D² for non-trivial
  post-stop systems.
* **Coord-break order swapped** (Zemax PARM 6).  The 0 / 1 branches
  for "decenter then tilt" vs "tilt then decenter" were inverted,
  so every imported folded design with the Zemax default got the
  wrong frame transform.
* **Lagrange invariant for finite-conjugate stop-at-front** —
  previously H was identically 0 (both y_m_init and y_c_init were 0),
  zeroing the Petzval contribution.  The marginal-ray initial height
  is now computed via T(d) ∘ M_pre.
* **Rayleigh-Sommerfeld kernel sign flipped** (``propagators/
  propagation.py``).  Goodman 3-43 gives ``(1/r − ik)``, code had
  ``(ik − 1/r)`` — every coherent superposition of RS with ASM /
  Fresnel was 180° out of phase.
* **``apply_fresnel_curvature`` half-pixel offset dropped** — was
  ``(arange − N/2 + 0.5) * dx``; every other propagator uses no
  offset.  Visible as a small coma residual in OPDPy cross-checks.
* **Tilted-ASM band-limit now centred on the original-frame spectrum
  ``FX + fx0``** — pre-4.10 the mask on plain ``|FX|`` killed the
  baseband DC and zeroed the propagated field for any non-trivial
  tilt with the default ``bandlimit=True``.
* **Fresnel / Fraunhofer honour complex64 input dtype** — pre-4.10
  silently promoted to complex128 via Python-float constants.
* **JAX trace correctness floor** (``raytrace/jax_trace.py``):
  ``_sag_derivatives_jax`` carries sign(R) for concave surfaces;
  Snell uses a double-where pattern at the TIR boundary so
  ``jax.grad`` is finite; aspheric Newton uses double-where on
  ``F/dF_dt`` for grazing rays; ``trace_jax`` raises
  ``NotImplementedError`` on mirror / coord-break / biconic /
  freeform surfaces instead of silently treating them as flat
  refractive.
* **JAX ``propagate_through_system`` aperture branch fixed**
  (``system.py``).  Was reading ``elem.get('radius')`` while the
  NumPy path uses ``{shape, params}``; every working NumPy
  aperture spec was silently no-op'd in the JAX path.
* **Richards-Wolf** (``propagators/vector_diffraction.py``): the FFT
  fixes ``dx_focal = λf/(N·dx_pupil)`` regardless of caller value
  (warning emitted if mismatched).  Adds the missing 1/√(cos θ)
  Jacobian (was using cos^(3/2) instead of cos^(-1/2) apodisation)
  and the Richards-Wolf prefactor ``−i k f/(2π) · exp(−i k f)``.
* **Asymptotic ``aberration_tensor``**: warns when the closed-form
  ℓ=0 path is used with multiple radial orders p — those collapse
  to the same scalar weight under the Wick contraction.  Pass at
  least one ℓ ≠ 0 to force the σ-grid path.  Also: HF Chebyshev
  quadrature multiplies by the missing ``−i`` Maslov prefactor
  so phase matches Fresnel; ``decompose_lg`` / ``decompose_hg``
  accept 1-D coordinate axes.
* **GBD reconstruction** now includes the per-beamlet tilt phase
  ramp ``exp(i k (L Δx + M Δy))`` so off-chief-ray interference
  patterns and PSF wings reconstruct correctly.
* **Optimize merits**:
  - ``evaluate()`` out-of-range-BFL branch returns ``(value, ctx)``
    tuple instead of bare scalar (was ``TypeError`` mid-run).
  - ``MultiWavelengthMerit`` now re-evaluates the wave leg at each
    wavelength; pre-4.10 it averaged the same single-wavelength
    field N times so chromatic merits were a no-op.
  - ``MultiFieldMerit`` aperture-masks the tilted plane wave (was
    grid-filling, biasing Strehl numbers).
  - ``MinBackFocalLengthMerit`` / ``MaxFNumberMerit`` guard the
    BFL = 1e9 sentinel via ``ctx_is_valid``.
  - ``design_optimize`` uses a ``__del__``-based dtype guard that
    fires even on scipy raise / KeyboardInterrupt (was leaking
    complex64 globally to the rest of the process).
* **Polarization**: ``create_circular_polarized('right')`` now
  returns ``(1, −i)/√2`` (RHC under exp(−iωt)).  Pre-4.10 it
  returned ``(1, +i)/√2`` (LHC); ``apply_waveplate`` /
  ``stokes_parameters`` were already on the correct convention so
  the round-trip was internally broken.
* **``mutual_coherence``**: returns ``<E(x_i) conj(E(x_j))>`` as
  documented (pre-4.10 returned the complex conjugate; Hermiticity
  preserved so the bug was silent).
* **Ghost TIS**: dropped the extra ``cos(θ)`` factor under
  cos-weighted hemisphere sampling.
* **Interferometry fringe formula**: now ``background * (1 +
  visibility * cos(phase))`` so visibility=1 gives full contrast.
* **Field aberration sweep**: ``field_aberration_sweep`` now builds
  true sagittal / tangential fans at a +y field (pre-4.10 used two
  *different* field directions and called them sag/tan).
  ``petzval_radius`` returns ``−1/inv_R`` (Born & Wolf §4.4).
* **Shack-Hartmann wavefront**: dropped the spurious
  ``wavelength/(2π)`` factor (cumsum of slopes is already in
  meters); both row- and column-integrals anchored to a common
  origin before averaging.
* **AO ``zernike_modal_basis``**: one-sided FD at the pupil rim
  (was generating spurious spikes from ρ > 1 evaluations); slopes
  divided by ``semi_aperture`` for physical-units consistency;
  ``pinv`` regularised at ``rcond=1e-3``.
* **Phase retrieval (JAX) error_reduction**: reordered to
  FFT → magnitude → IFFT → support, matching NumPy.
* **Tolerancing reproducibility**: replaced ``hash(knob)`` (Python-
  3 process-randomised) with a deterministic knob-to-int map.
* **Through-focus JAX backend now matches NumPy** for
  ``rms_radius`` (D4σ about centroid) and ``power_in_bucket``
  (absolute integrated intensity, not fraction).
* **Real-lens (``elements/_lens_real.py``)**: Fresnel unpolarised
  scalar throughput uses intensity averaging
  ``√(0.5(|t_s|² + |t_p|²))``; Seidel correction drops the spurious
  negation that doubled the applied polynomial; ``np.gradient``
  honours ``dy`` for anamorphic grids.
* **Thin-lens (``elements/_lens_thin.py``)**: ``apply_axicon``
  imports ``get_glass_index`` (was missing → ``NameError`` for
  string glass); aplanatic phase mask uses ``1+0j`` outside the
  valid domain so the rim aperture mask alone controls amplitude;
  ``apply_aspheric_lens`` NaN-propagates outside the conic domain.
* **Coatings**: T computed via amplitude transmission t (Macleod
  eq. 2.99) so absorbing stacks correctly give R + T < 1; intra-
  stack TIR emits a warning instead of silently capping.  Dead
  ``num``/``den`` code removed.
* **Conic sag** at the edge: ``surface_sag_general`` returns NaN
  outside the conic domain instead of silently 0 (so aperture
  masks zero those pixels deterministically).
* **DOE FZP**: ``create_fresnel_zone_plate`` raises on
  ``focal_length ≤ 0`` (was silently stripping the sign).
* **Glass**: ``_sellmeier_index`` validates against Sellmeier
  resonances and refuses negative radicands instead of raising
  opaque ``math domain error``.
* **``user_library.load_phase_mask``**: requires explicit
  wavelength (was silently defaulting to 1.0 m → useless phase);
  ``eval()`` rejects expressions with ``__`` or leading ``_``
  tokens as defence-in-depth.
* **Top-hat / annular / Bessel** source factories now accept ``dy``
  (were hard-coded to ``dy = dx``).
* **Source generators ``use_gpu=True``** now calls
  ``_ensure_cupy_loaded()`` (was ``AttributeError`` on first GPU call).

### Behavioural changes that may need caller-side adjustments

* ``richards_wolf_focus(dx_focal=...)`` now emits a ``RuntimeWarning``
  if the caller-supplied value differs from the FFT-natural pitch;
  the FFT pitch is used either way.
* ``thin_film_stack`` reports physical absorptive ``T`` for lossy
  stacks; if you were relying on ``T = 1 − R`` for an absorbing
  multilayer, you'll see lower transmission than before.
* ``apply_real_lens(..., fresnel=True)`` for unpolarised input
  now uses intensity-mean transmission, slightly different from
  the pre-4.10 amplitude-mean approximation.
* ``apply_axicon('SOME_GLASS', ...)`` no longer raises NameError.
* ``MultiWavelengthMerit`` actually evaluates chromatic Strehl /
  RMS-OPD merits — if you had a "tuned" set of weights, expect the
  optimal point to move now that the merit isn't a no-op.

### Methodology notes

This release was driven by three concurrent audit runs (one
external 8-agent run, two internal 5-agent multi-agent audits).
Findings were cross-checked across the three reports; each
substantive bug appears in ``CORRECTION_PLAN.md`` with a
file:line citation.  The full validation suite (34 files, 314
tests) was run after every group of fixes; a small number of
tests were updated to reflect the new behaviour (e.g.
``apply_fresnel_curvature`` half-pixel offset removal).

## [4.9.0] — 2026-05-15

**Bundled 4.8.1 + external-audit response.**  4.9 ships the scoped
runtime-environment manager planned for 4.8.1 alongside the
correctness fixes from the v4.8.0 external audit
(``LumenAiry_Audit_Report.md``).  All 7 verified physics bugs and the
2 documentation gaps the audit flagged are closed; the high-impact
Seidel formula correction (#2.1) cascades through Petzval (#4.6),
Schwarzschild S5 (#4.7), and the optimization-hot-loop merit terms.
170 unit tests (+ 16 new audit-regression tests, + 29 new
context-manager tests, + 6 new ground-truth Seidel tests) and all 34
validation files pass.

### Scoped runtime-environment (4.8.1 work bundled in)

* **``lumenairy_context``** -- a new context manager that snapshots
  and restores the library's process-global runtime settings
  (``complex_dtype``, ``pyfftw_planner``, ``fft_threads``,
  ``max_ram``, ASM cache caps) for the duration of a ``with``
  block.  Nests cleanly, restores on exception, optional
  ``clear_caches_on_exit=True`` for hard experiment isolation.
* **``dtype=`` kwarg on the 11 source factories** -- explicit
  per-call dtype control: ``create_gaussian_beam(...,
  dtype=np.complex64)`` allocates a complex64 field regardless of
  ``DEFAULT_COMPLEX_DTYPE``.  Default ``dtype=None`` inherits from
  the library default (4.9 also fixes the pre-existing
  inconsistency where factories silently returned complex128
  regardless of the global default).
* **``atexit`` auto-restore** -- on first ``import lumenairy`` the
  default runtime state is snapshotted and an atexit handler
  restores it on process shutdown.  Catches the foot-gun where
  ``set_default_complex_dtype`` / ``set_pyfftw_planner`` /etc.
  called at module scope in a long-running session would otherwise
  permanently leak state.
* **New getters** to support the round-trip:
  ``get_pyfftw_planner``, ``get_fft_threads``, ``get_asm_cache_size``,
  ``get_max_ram``.

### Audit-fix physics bugs

* **#2.1 Seidel coefficient formula** -- ``seidel_coefficients`` was
  using ``Δ(1/n) = 1/n2 − 1/n1`` where Welford's per-surface formula
  requires ``Δ(u/n) = u_after/n2 − u_before/n1``.  The two differ by
  a surface-geometry-dependent factor (typically 1.5×-5× per surface);
  the buggy magnitudes propagated through ``seidel_wfe``,
  ``seidel_field_sweep``, and ``SphericalSeidelMerit``.  Fix: use
  the correct Welford ``Δ(u/n)``.  4.9 also fixes the flat-refracting-
  surface branch which previously zeroed S1/S2/S3 (a plano-convex
  singlet's flat back surface actually contributes nonzero S1 because
  ``Δ(u/n) ≠ 0`` at non-normal incidence).
* **#2.5 ``aberration_tensor`` ℓ ≠ 0 outputs** -- the pre-4.9 chief-ray
  projection collapsed to the constant term of the LG output polynomial,
  which is identically zero for any ℓ ≠ 0 mode ((σ_x + j·σ_y)^|ℓ| has
  no constant term).  Coma ``(1, ±1)``, astigmatism ``(0, ±2)``, tilt
  ``(0, ±1)``, and every other ℓ ≠ 0 output silently returned 0.  4.9
  implements the full output-plane σ-integration via
  ``propagate_modal_asymptotic`` + ``decompose_lg`` on a small grid
  around the chief image -- ℓ ≠ 0 modes now carry real physical
  meaning.  ℓ = 0 outputs keep the fast closed-form chief-ray path.
  New ``sigma_grid_n`` / ``sigma_grid_extent`` kwargs tune the
  projection accuracy.
* **#4.6 / #4.7 ``seidel_wfe`` Petzval H²** -- the WFE expansion
  used ``S4·sigma²·ρ²`` where Welford requires
  ``S4·|H|²·ρ²`` (Lagrange invariant squared).  For a 100 mm BK7
  singlet at f/4 this was off by ``(D/2)² = 1.6e-4 m²``,
  producing ~100 mm of phantom Petzval WFE.  ``seidel_coefficients``
  now returns ``'lagrange_invariant'`` in its result dict, and
  ``seidel_wfe`` uses |H|² for the S4 term.  The Schwarzschild
  relation ``S5 = −(A_c/A_m)·(S3 + H²·S4)`` also picks up the
  missing H² factor for dimensional consistency.
* **#2.2 GBD back-prop sign** -- ``propagate_beamlets_freespace``
  used ``exp(1j·k·abs(t))``, which complex-conjugates the axial
  phase on back-propagation.  Fix: ``exp(1j·k·t)`` (signed).
  Forward propagation was unaffected; back-prop now round-trips.
* **#2.3 Coronagraph λ/D scaling** --
  ``coronagraph_contrast_curve`` hard-coded
  ``pix_per_lam_over_D = float(Nx)``, ignoring the
  ``wavelength``, ``f_eff``, ``dx_focal`` kwargs it accepted.
  Only correct for the FFT-natural pitch (no zero-padding); for
  oversampled / MFT-zoomed focal grids the angular-scale label
  was wrong.  4.9 adds ``pupil_diameter_m`` and computes
  ``pix_per_lam_over_D = λ · f_eff / (D · dx_focal)``.  Legacy
  callers without ``pupil_diameter_m`` get a one-time
  ``RuntimeWarning`` and the pre-4.9 approximation.
* **#3.3 Fresnel / Fraunhofer / SAS z ≤ 0 guards** -- these
  propagators are forward-only by construction; pre-4.9 silently
  produced garbage on ``z ≤ 0``.  4.9 raises ``ValueError`` with
  a pointer to ASM / RS (which do handle back-propagation).
* **#3.5 TIR mask placement** -- the total-internal-reflection
  mask in ``apply_real_lens`` was nested inside ``if fresnel:``,
  so callers using ``slant_correction=True`` with ``fresnel=False``
  got unphysical residual field in TIR regions.  4.9 runs the TIR
  mask whenever either path computes ``sin2_tt``.
* **#4.5 Cosmic-ray rate scaling** -- ``apply_detector``'s
  ``cosmic_ray_rate`` kwarg was a single Poisson parameter for
  the whole exposure, not scaled by detector area · exposure
  time as the physics demands.  4.9 adds
  ``cosmic_ray_rate_per_m2_per_s`` for physically-correct
  scaling; legacy ``cosmic_ray_rate`` emits a
  ``DeprecationWarning`` when used.

### Audit-fix small items

* **#4.3 ``dx > 1 mm`` validator loosened.**  Pre-4.9 raised on
  ``dx > 1 mm`` (incorrectly rejecting large-aperture telescope
  pupils at mm-scale sampling).  4.9 raises only on ``dx > 100 mm``;
  the 1-100 mm range warns once but proceeds.
* **#4.4 Zemax INCH / INCHES alias.**  The ``load_zemax_zmx`` unit
  map gained INCH and INCHES aliases alongside IN.

### Audit-fix documentation

* **#2.4** -- ``coating_reflectance`` docstring now calls out the
  Snell ``.real`` simplification (wrong for metal layers) and the
  silent TIR cap inside the stack.
* **#3.1** -- ``propagate_hfpi`` docstring documents the missing
  ``1/(jλ)``, ``1/r``, and Monte Carlo normalization factors;
  reframes as a phase-structure / interference diagnostic, not a
  quantitative-amplitude propagator.
* **#3.2** -- ``decompose_field_to_beamlets`` docstring documents
  the position-only limitation (tilted-input fields are
  reconstructed at the output via phase ramps, not direction shifts).
* **#3.4** -- ``apply_real_lens`` scalar Fresnel transmission noted
  as valid only for 45° linear polarisation at modest AOI.
* **#4.2** -- ``lg_polynomial`` docstring now includes the ``/ w``
  factor in the LG normalisation (the code always carried it
  correctly).

### Regression tests added

* ``tests/unit/test_context_manager.py`` -- 29 tests for the new
  context manager, factory ``dtype=`` kwarg, and atexit hook.
* ``tests/unit/test_seidel_ground_truth.py`` -- 6 tests for the
  Seidel formula fix, including the ground-truth ray-trace
  vs OPD-fit comparison the audit recommended.
* ``tests/unit/test_audit_fixes_v4_9.py`` -- 16 tests covering
  each verified audit finding (GBD back-prop, coronagraph scaling,
  Fresnel z<=0 guards, TIR mask, dx validator, Zemax INCH alias,
  cosmic-ray scaling).

---

## [4.8.0] — 2026-05-15

**Library-correctness pass triggered by the external wiki audit.**
Fixes three verified bugs surfaced by the audit and closes the two
long-deferred items (`DeformableMirror._IF_basis` memory foot-gun
from the 4.0.1 deferred list; folded-design wave-optics silent-drop
from the 3.7.8 archive page).  119 unit tests + 34 validation files
pass.

### Bug fixes

* **`create_point_source` sign convention.**  Formula was
  `E = A * exp(+i*k*r) / r` for both signs of `z0`, always producing
  a diverging wave.  Fixed: `z0 <= 0` → `exp(+i*k*r)/r` (diverging,
  source before grid); `z0 > 0` → `exp(-i*k*r)/r` (converging, focus
  after grid).  Code that used `z0 > 0` to model an *outgoing* wave
  must flip to `z0 < 0`.
* **`FocalLengthMerit` / `BackFocalLengthMerit` afocal target.**
  The `target == 0` branch returned `weight * efl**2`, growing
  without bound as `efl → ∞` instead of pushing toward infinity as
  the docstring promised.  Switched to penalising optical power
  `(1/efl)^2`; merit → 0 as `efl → ∞`.

### Long-deferred items now fixed

* **`DeformableMirror._IF_basis` memory foot-gun** (deferred since
  4.0.1 audit).  Eager `(n_act, n_act, N, N)` pre-allocation was 8
  GB float64 for a 32×32 actuators × 1024×1024 pupil.  New
  `cache_basis = {'auto', True, False}` flag (default `'auto'`)
  caches eagerly below a 512 MB ceiling and streams on demand past
  that.  New `DeformableMirror.fit_phase(target_phase)` public
  modal-to-zonal projection helper.
* **Wave optics through folded designs silent-drop.**
  `apply_real_lens`, `apply_real_lens_traced`, and
  `apply_real_lens_maslov` silently ignored mirrors in
  `prescription['elements']`; for a folded `.zmx` this propagated
  the unfolded-equivalent path with the mirror's curvature phase
  and world-frame axis change dropped.  All three entry points now
  raise `ValueError` unless the caller acknowledges via
  `prescription['allow_unfolded_equivalent'] = True`.  New public
  helpers `split_prescription_at_mirrors(rx)` and `has_mirrors(rx)`
  support the segment-by-segment workflow (alternate
  `apply_real_lens` with `apply_mirror` at each fold).

### Wiki audit closeout

The companion wiki underwent a comprehensive review pass: all 20
critical, all 58 major, and ~290 of 294 medium findings addressed
across 18 commits.  Highlights: ASM aliasing thresholds documented
in Tutorial / Quickstart with `check_sampling_conditions` calls
made explicit; test-count contradictions reconciled (34 files /
~700 tests / 119 unit); Physics XII §1 / §5 photon-noise formulas
corrected (`C_lim` non-inverted); Marechal-Strehl exponent fixed
(`S ≈ exp(-σ_φ²)`, not `exp(-2σ_φ²)`); Fresnel-number aliasing
criterion corrected from `N²/4` to `N/4`; new FR-Memory page
documenting RAM + FFT-planner helpers; FR-Storage `replay_run`
pre-3.7.x archive caveat; FR-Asymptotic `LGAberrationMerit`
(±ell vortex orientations vs sin/cos clarification);
Validation L8/L9 absolute-residual-scaling explainer.

### Regression tests added

* `tests/unit/test_optimize_merit_terms.py` — 5 tests covering
  the afocal-target branch.
* `tests/unit/test_ao_dm.py` — 9 tests covering DM cache modes
  and `fit_phase` round-trip.
* `tests/unit/test_folded_design_guard.py` — 9 tests covering
  the silent-drop guard, `allow_unfolded_equivalent` escape hatch,
  and `split_prescription_at_mirrors` / `has_mirrors` helpers.
* `tests/unit/test_sources.py` augmented with point-source
  sign-convention round-trips.

---

## [4.7.0] — 2026-05-14

**Polish-pass release + API-consistency pass.**  Implements the verified
items from ``lumenairy_4_6_polish_pass.md`` (A.1, A.2, A.3, B.1, B.2,
B.4, B.5, B.6, B.7, B.9, B.11, C.1, C.2, D, F.1, F.3, F.4, H.1, H.2,
H.3, H.4, H.5, I.1, I.2, I.3).  All 34 validation files (~670 ``t_*``
assertions) and 94 unit tests pass.

### Breaking changes (single-user library — no deprecation cycle)

The 4.7 release intentionally breaks several positional / kwarg
conventions to remove footguns that the polish-pass flagged.  Existing
external callers (none known at the time of release) must migrate; the
4.6 -> 4.7 changes are uniform and mechanical.

* **B.1 -- lens-function args are now keyword-only past ``E_in``.**
  ``apply_thin_lens``, ``apply_spherical_lens``, ``apply_aspheric_lens``,
  ``apply_cylindrical_lens``, ``apply_grin_lens``, ``apply_real_lens``,
  ``apply_real_lens_traced``, ``apply_real_lens_maslov``, and their
  ``_jax`` twins.  This removes the positional inconsistency where
  ``wavelength`` sat at position 3 in some, 5 in others, and 6 in
  ``apply_spherical_lens`` -- a copy-paste typo could silently swap
  ``wavelength`` and ``dx``.  Pass everything as keyword arguments::

      apply_real_lens(E, prescription=p, wavelength=lam, dx=dx)
      apply_thin_lens(E, f=0.05, wavelength=lam, dx=dx)
      apply_spherical_lens(E, R1=..., R2=..., d=..., n_lens=...,
                            wavelength=lam, dx=dx)

* **B.2 -- ``lens_prescription=`` kwarg renamed to ``prescription=``**
  on the real-lens trio.  Matches the rest of the library (54 prior
  uses of ``prescription``).  The 4.6 alias is removed.

* **B.5 -- diffractive-lens factories dropped the ``_m`` suffix.**
  ``create_diffractive_lens``, ``create_kinoform``,
  ``create_fresnel_zone_plate`` now take ``dx`` / ``focal_length`` /
  ``wavelength`` (no ``_m`` suffix).  LumenAiry uses SI metres
  throughout, so the suffix was redundant.

* **B.6 -- source-factory ordering standardised on
  ``(N, dx, wavelength, *, source_specific)``.**  Affected functions:
  ``create_gaussian_beam`` (``sigma`` is now kwonly),
  ``create_top_hat_beam`` (``diameter`` kwonly),
  ``create_annular_beam`` (``outer_diameter`` / ``inner_diameter``
  kwonly), ``create_fiber_mode`` (``mode_field_diameter`` kwonly).
  Old: ``create_gaussian_beam(N, dx, sigma, wavelength=lam)``.
  New: ``create_gaussian_beam(N, dx, lam, sigma=sigma)``.

* **B.7 -- ``wavelength`` is now required (no default) on**
  ``keplerian_telescope``, ``beam_expander_prescription``,
  ``export_zemax_lens_data``, ``export_zemax_zmx``, ``export_codev_seq``,
  ``export_quadoa_qos``, and ``make_ray``.  Removes the ``550e-9`` /
  ``1.31e-6`` defaults that disagreed across the library.

* **C.2 -- Zemax loader rename.**  ``load_zmx_prescription`` →
  ``load_zemax_zmx`` (matches ``export_zemax_zmx``).
  ``load_zemax_prescription_txt`` → ``load_zemax_prescription_data_txt``
  (disambiguates from the ``.zmx`` loader).

### Added (non-breaking)

### Added

* **Propagator input validation.** A new private helper
  ``_validate_propagator_inputs(E, z, wavelength, dx, dy=None)`` is
  called at the entry of every public free-space propagator (ASM,
  tilted ASM, batch ASM, MFT ASM, Fresnel, Fresnel-MFT, Fraunhofer,
  Fraunhofer-MFT, Rayleigh-Sommerfeld, scalable ASM).  It catches the
  silent-failure regimes the old code shipped with:
  ``wavelength = 0`` / negative (``ZeroDivisionError``);
  ``wavelength = 1.31`` (forgot the e-6, silent garbage);
  ``dx = 0`` (``ZeroDivisionError``);
  ``dx = 2.0`` (forgot units, silent garbage);
  3-D / 1-D / empty / NaN inputs; non-finite ``z``.  Errors quote the
  parameter name, the offending value, and the calling function.

* **Prescription input validation.** New public
  ``lumenairy.validate_prescription(prescription, *, strict=True)``
  helper, and ``surfaces_from_prescription`` now calls it internally
  before any conversion.  Catches: empty dict, missing
  ``'surfaces'`` / ``'thicknesses'``, surface count / thickness
  length mismatch, NaN radius, missing glass keys, non-positive /
  non-finite ``aperture_diameter``.  ``strict=False`` returns the
  issue list instead of raising.

* **Glass registry rebuild.**

  * ``GLASS_REGISTRY`` entries can now be a *callable*
    ``f(wavelength_m) -> n`` (returning ``float`` or ``complex``).
    Users register custom dispersion models, prototype coatings, or
    temperature-dependent indices with a one-line lambda.
  * New bundled ``SELLMEIER_COEFFICIENTS`` table with ~30 Schott /
    Ohara entries (N-FK51A, N-PSK53A, N-LAK33A/B, N-SK11, N-SK16,
    N-SSK2, N-SF5/10/11/14/15/57, N-LASF31A/40/41/44/45/46A/46B,
    F2, F5, SF2, S-LAH64, S-LAH79, BaF2, ...).  These work as a
    no-external-deps fallback when the optional
    ``refractiveindex`` package is missing.  Sellmeier output
    matches the live refractiveindex.info lookup to machine
    precision for the overlapping entries.
  * New ``list_glasses()`` and ``search_glasses(pattern)`` helpers
    for discoverability.  Errors on unknown glass names now include
    a ``Did you mean: [...]`` suggestion (substring match first,
    difflib closest-spelling match as fallback).

* **``dy`` kwarg on the lens trio.**  ``apply_real_lens``,
  ``apply_real_lens_traced``, and ``apply_real_lens_maslov`` now
  accept ``dy=None`` for API symmetry with the rest of the lens
  family.  ``apply_real_lens`` propagates non-square pixels through
  the per-surface phase screens and the in-glass ASM.  The traced
  and Maslov variants accept the kwarg but raise on ``dy != dx``
  (their interpolation paths assume square pixels).

* **Field-analysis dataclass returns.**  The 4.4-era
  ``distortion_grid``, ``footprint_per_surface``, and
  ``spot_diagram_vs_field`` returned bare dicts / lists of dicts,
  while their siblings (``distortion_vs_field``,
  ``relative_illumination``, ``field_aberration_sweep``,
  ``sensitivity_ranking``) already returned named dataclasses.  This
  inconsistency is now resolved: new ``DistortionGrid``,
  ``SurfaceFootprint``, ``FieldFootprint``, and ``SpotDiagramField``
  dataclasses replace the dict returns.  They subscript like dicts
  (``result['actual_x']`` and ``fp[0]['fields'][0]['x']`` still
  work) so 4.4 callers keep working.

* **Module-level ``__all__``** in all 9 analysis submodules
  (``core``, ``coherence``, ``detector``, ``ghost``,
  ``image_plane_wfe``, ``through_focus``, ``interferometry``,
  ``phase_retrieval``, ``plotting``).  ``from lumenairy.analysis.core
  import *`` now exports exactly the documented public surface.

* **PyPI ``Changelog`` + ``Releases`` URLs** in ``pyproject.toml``
  so the project page on PyPI links directly to the in-repo
  ``CHANGELOG.md`` and the GitHub Releases page.

* **A.3 -- ``apply_real_lens`` kwarg-combination validation.**  A new
  ``_check_apply_real_lens_kwarg_combination`` guard runs at the top
  of ``apply_real_lens`` and rejects nonsensical combinations
  (unknown ``wave_propagator`` value; ``slant_correction=True`` with
  Fresnel propagator; ``seidel_correction=True`` on a 1-surface
  prescription; non-positive / out-of-range ``seidel_poly_order``).

* **H.1 -- propagator-trio canonical-order aliases.**  New
  ``propagate_gbd(E, z, wavelength, dx, ...)`` (already shipped in
  4.6), ``propagate_hfpi(E, z, wavelength, dx, ...)``, and
  ``propagate_huygens_fresnel(E, z, wavelength, dx, ...)`` provide a
  single canonical entry point per family that matches
  ``angular_spectrum_propagate``'s argument order.  The legacy
  per-leg functions (``propagate_*_freespace``, ``*_thin_lens``,
  ``*_through_prescription``, ``*_with_opl_callable``) remain for
  specialised use cases.

* **H.2 + H.3 -- ASM auto-selector and advisor.**  New
  ``asm_propagate(E, z, wavelength, dx, **kw)`` picks between
  ``asm`` / ``asm_tilted`` / ``asm_mft`` / ``sas`` / ``fraunhofer``
  based on the geometry (tilt, requested output pitch, Fresnel
  number) and runs the chosen one.  ``which_propagator(...)``
  returns the same choice *without* running, surfacing the decision
  for documentation / GUI display.

* **H.4 -- ``plot_lens_layout``.**  Script-/notebook-callable 2-D
  cross-section drawing of a lens prescription, lifted out of the
  GUI's ``Layout2DView``.  Renders surfaces, glass shading, optical
  axis, image plane, and (optionally) traced ray fans.  Independent
  of Qt; needs only matplotlib.

* **H.5 -- ``plot_glass_map`` + ``abbe_diagram``.**  Classic n_d-vs-
  V_d Abbe-diagram scatter (with annotations) and a two-panel
  variant that pairs the Abbe diagram with full ``n(lambda)``
  dispersion curves over the visible + near-IR.  Lifted out of
  the GUI's ``glass_map_dock``.

* **D -- ``lumenairy._deprecation`` helper module.**  Centralises
  ``DeprecationWarning`` emission across the polish-pass work so
  future deprecations have a uniform message format.  Public
  helpers: ``warn_deprecated_kwarg``, ``warn_deprecated_alias``,
  ``deprecated_alias``, ``warn_deprecated_default``.

* **F.4 -- ``py.typed`` marker.**  Empty ``lumenairy/py.typed`` file
  registered in package-data so type checkers (mypy, pyright,
  pylance) treat the library's type hints as authoritative rather
  than skipping them.

* **C.4 -- ``backend='jax'`` kwarg consolidation.**  The
  ``gerchberg_saxton``, ``error_reduction``, ``hybrid_input_output``,
  ``through_focus_scan``, and ``monte_carlo_tolerancing`` functions
  now accept a ``backend='numpy' | 'jax'`` kwarg that dispatches to
  the underlying JAX-traced implementation when requested.  The
  ``_jax``-suffixed siblings remain importable for low-level use,
  but ``backend='jax'`` is the canonical entry point.  Lens-trio
  JAX twins (``apply_real_lens_traced_jax``,
  ``apply_real_lens_maslov_jax``) and ``trace_jax`` stay as
  separate entry points because their semantically-distinct
  autodiff paths benefit from the explicit name.

* **E -- Type-annotation coverage push.**  Annotated ~490 public
  functions across propagators, elements, raytrace, sources,
  analysis, io, optimize, glass, system, and backend.  Coverage
  jumped from **10.5% fully typed (28.5% any) → 70.1% fully typed
  (90.2% any)**, exceeding the polish-pass 80% target.  Mypy /
  pyright / pylance now have authoritative type information for
  the entire user-facing API.  Forward references in
  ``raytrace/bundles.py`` use ``TYPE_CHECKING`` imports to break
  the runtime circular dependency.

* **G -- ``pytest validation/`` works natively.**  Added
  ``validation/conftest.py`` that wires pytest collection for the
  670 existing ``t_*`` functions and converts their
  ``(bool, str)`` returns into real ``AssertionError`` raises.
  ``pyproject.toml`` now lists ``t_*`` alongside ``test_*`` in
  ``python_functions``.  Auto-marks every collected ``t_*`` as
  ``integration`` so it respects the project-wide
  ``-m "not integration"`` default.  Legacy
  ``python validation/run_all.py`` driver works unchanged.  New
  path enables IDE test discovery, ``-k`` filtering, JUnit XML,
  coverage, and parallel execution via ``pytest-xdist``.

* **B.10 -- Array-namespace dispatch was already in place.**
  Audit confirmed that 55 sites already use
  ``array_namespace(E_in)`` dispatch (or equivalent
  ``_is_cupy_array(E_in)`` / ``is_jax_array(E_in)`` checks) to
  automatically route through NumPy / CuPy / JAX based on the
  input array type.  ``use_gpu=True`` remains as the back-compat
  opt-in for forcing NumPy → CuPy promotion.  README updated to
  document the convention explicitly.

### Renamed

* **``lumenairy.analysis.analysis`` → ``lumenairy.analysis.core``.**
  The historical doubled-name was an accident.  A back-compat shim
  at ``lumenairy/analysis/analysis.py`` re-exports the new module's
  public surface, so existing user code that did
  ``from lumenairy.analysis.analysis import beam_centroid``
  keeps working.  New code should use
  ``from lumenairy.analysis import beam_centroid`` (top-level) or
  ``from lumenairy.analysis.core import beam_centroid`` (explicit).

### Updated

* **``ROADMAP.md``** -- header rewritten to reflect current 4.7.0
  state (was still saying "weaknesses in `lumenairy` v3.0").  Added
  a "Resolved since 3.0" appendix summarising the 4.x feature
  arrivals.  Section 11 validation counts updated from the old
  21-case OPD claim to the current ~670 ``t_*`` assertions across
  34 files + 49 unit tests + multi-platform CI.

* **Internal callers** for the renamed functions and signature
  changes were updated across 40+ files in ``validation/``,
  ``tests/``, ``examples/``, ``lumenairy/ui/``, and the library
  proper.  Three pre-existing positional-order bugs surfaced and
  were fixed as a side effect of the rewrite: a swapped ``dx`` /
  ``f`` in ``examples/01_basic_propagation.py``, the same in
  ``validation/raytrace/test_raytrace.py``, and a swapped
  ``dx, wavelength`` in ``lumenairy/propagators/dispatch.py``.

## [4.6.0] — 2026-05-14

**Documentation overhaul -- decision-tree front door + lens-family
cross-refs.**  No code changes; the public API is identical to 4.5.0.
The 33 validation files and 52 unit tests still pass unchanged.

### Changed (documentation only)

* **README rewritten around "which function should I use?"** -- the
  former Quick Start section was replaced with five "if you need X,
  use Y" tables covering free-space propagation, lens application
  (the three `apply_real_lens*` variants), folded designs, output
  field analysis, and design optimisation.  Three minimal end-to-end
  examples follow.  The longer recipes (Zernike decomposition,
  polarization, phase retrieval, HDF5 storage, plotting, etc.) moved
  into a new `## Cookbook` section near the end of the README, where
  users who need them can still find them.

* **`apply_real_lens` / `_traced` / `_maslov` docstrings cross-link
  to each other.**  Each now opens with a `See Also` block + a
  one-line `Quick decision guide` so the choice between the three
  fidelity points is visible at the top of `help(la.apply_real_lens)`.

* **Dense physics citations moved out of the README main flow.**
  Matsushima-Shimobaba and Heintzmann-Loetgering-Wechsler kernel
  details, per-variant real-lens accuracy notes, and the Maslov
  integral specification now live in the renamed `Appendix: Physics
  references` section at the bottom of the README.  The dense
  per-`apply_real_lens` variant paragraphs in `Key Features` were
  trimmed to one-line summaries that point at the docstrings + the
  appendix.

### No code changes

The public API, all imports, all exports, and the wheel contents
are identical to 4.5.0.

## [4.5.0] — 2026-05-13

**World-frame ray tracing for folded prescriptions, end-to-end.**
Closes the deferred 4.4 gap: a folded prescription can now go from
``load_zmx_prescription`` to ``trace_world`` and to a world-coordinate
paraxial focus, all without instantiating the GUI's ``SystemModel``.
Pure-additive; no breaking changes.

### Added

* **`paraxial_focus_world(world_surfaces, wavelength, *,
  aperture_radius=None)`** in `lumenairy.raytrace.world` -- returns
  the world-frame ``(focus_origin, focus_normal)`` of the paraxial
  image plane.  Traces a chief + paraxial-marginal ray with
  ``trace_world`` and finds the closest-approach point of the two
  ray lines, which is robust to mirror reflections and Zemax-signed
  post-mirror thicknesses (the failure mode of a direct BFL-along-
  world-axis approach).

### Changed (additive, backward-compatible)

* **`surfaces_from_prescription` honours the per-surface
  ``is_mirror`` flag and the ``glass_after='MIRROR'`` Zemax marker
  string.**  Previously the flag was silently dropped, so a folded
  prescription's mirrors became refractive (and ``trace_world`` then
  failed on the bogus ``'MIRROR'`` glass lookup).  4.5 auto-infers
  ``is_mirror=True`` and normalises ``glass_after`` to
  ``glass_before`` so the surface plumbs correctly through
  ``trace_world``.

  Affects any hand-built or `.zmx`-loaded folded prescription;
  un-folded prescriptions are unaffected.

### Validation

* 1 new validation file: `validation/raytrace/test_folded_designs.py`
  (8 tests).  Builds two folded designs from scratch -- a periscope
  (plano-convex singlet + 45-deg flat fold mirror) and an oblique
  concave spherical mirror at 45 deg -- and cross-checks each result
  against an independent analytical baseline:
  * mirror + detector land at the correct world coordinates;
  * an on-axis chief ray traces through the fold and lands at the
    detector vertex (sub-nanometre agreement);
  * the singlet's paraxial focus matches the analytical
    ``BFL + last-surface-z``;
  * the curved fold mirror focuses to the analytical Coddington
    tangential focal length ``f_t = R/2 cos(45 deg) ~ 70.7mm`` in
    +y post-fold;
  * straight-axis prescriptions give a world focus identical to
    ``last_surface_z + BFL``.

  These analytical cross-checks (singlet thin-lens image position;
  closed-form oblique-spherical-mirror tangential focal length)
  are an independent ground truth equivalent to an external
  library cross-check on the same problems.

* All 33 pre-existing validation files still pass after the
  ``surfaces_from_prescription`` change.

## [4.4.0] — 2026-05-13

**Field-resolved analyses lifted from GUI + world-frame builder for
folded prescriptions + Strehl alternatives + loud glass-catalog
failures + grating rename + packaging fix.**  Mostly additive; one
breaking name change is called out below.

### Added — World-frame surface builder

* **`world_surfaces_from_prescription(prescription)`** in new
  `lumenairy.raytrace.world` -- translates a prescription with
  COORDBRK entries (loaded via `load_zmx_prescription` from a folded
  `.zmx`) into a list of `Surface` objects with `world_origin` (m)
  and `world_R` (3×3) populated.  Pair with `trace_world` to trace
  folded designs from any script, without instantiating the GUI's
  `SystemModel`.  Honours Zemax PARM convention (tilt order, decenter
  ordering via PARM 6).
* **`trace_world` exported at top level**: `la.trace_world(rays,
  world_surfaces, wavelength)` for one-stop folded-design tracing.

Validation: 7 tests in `validation/raytrace/test_world_surfaces.py`
cover straight-axis equivalence with local trace (bit-identical
image-plane positions), tilt rotation correctness, decenter shift,
and Zemax PARM-6 order field.

### Added — Field-resolved analyses (new `lumenairy.analysis.field`)

Eight analyses that previously lived only inside ``ui/*_dock.py``
``_replot()`` bodies are now first-class public API, callable from any
script or notebook.  The GUI docks have been refactored to call these
public functions and render the returned dataclasses -- GUI behavior
is preserved bit-for-bit.

* **`distortion_vs_field(system, wavelength, max_field_deg, *,
  n_points=21)`** -- chief-ray f-tan(theta) distortion sweep.  Returns
  `DistortionVsField` dataclass.
* **`distortion_grid(system, wavelength, max_field_deg, *, n_grid=7)`**
  -- 2-D distortion grid (paraxial vs traced).
* **`footprint_per_surface(system, wavelength, *, semi_aperture,
  fields_deg, num_rings, rays_per_ring)`** -- per-surface ray
  footprints grouped by field.
* **`spot_diagram_vs_field(system, wavelength, fields_deg, *,
  semi_aperture, num_rings, rays_per_ring)`** -- spot diagrams at
  multiple field angles on a common image plane.
* **`sensitivity_ranking(merit_fn, x0, labels=None, *, eps_rel,
  eps_abs_floor)`** -- central-difference d(merit)/d(var) for an
  arbitrary parameter vector.  Returns `SensitivityResult`.
* **`relative_illumination(system, wavelength, fields_deg, *, ...)`**
  -- geometric vignetting fraction vs field (new).  Returns
  `RelativeIllumination`.
* **`field_aberration_sweep(system, wavelength, fields_deg, *, ...)`**
  -- real-ray sagittal / tangential focus shifts and astigmatism vs
  field.  Companion to the paraxial `seidel_field_sweep`.  Returns
  `FieldAberrationSweep`.
* **`petzval_radius(system, wavelength)`** -- paraxial Petzval
  surface radius from the curvature-only Hopkins sum.

All eight accept either a prescription dict or a pre-built surface
list.  Internal `_trace` dispatcher routes to `trace_world` when the
surface list carries world frames (folded designs) and `trace`
otherwise (standard / prescription-based designs).

### Added — Strehl alternatives in `lumenairy.analysis.analysis`

* **`strehl_marechal(rms_waves)`** -- closed-form
  `exp(-(2*pi*sigma)^2)` Strehl approximation from an RMS estimate.
  ~0.82 at 1/14 wave (the diffraction-limited rule of thumb).
* **`strehl_phase_integral(pupil)`** -- exact small-aberration Strehl
  `|<A exp(i*phi)>|^2 / <A>^2` (Born & Wolf 9.1.10).  Avoids the
  peak-finding bias of `strehl_ratio` on asymmetric PSFs.

### Changed

* **`aberration_summary` now warns loudly on unknown glass.**  When a
  prescription references a glass not in `GLASS_REGISTRY`, the
  function previously returned zero-filled Seidel coefficients with
  the error buried in `notes` -- making an unanalyzable system look
  diffraction-limited.  4.4 issues a `UserWarning` for the
  glass-lookup failure while keeping the zero-fill behavior for
  back-compat.

### Breaking change

* **`rcwa_1d` removed.**  The function's name advertised full
  Rigorous Coupled-Wave Analysis but the implementation has always
  been an analytical thin-grating scalar approximation (reflection
  hardcoded to zero, no S-matrix interface matching).  Call
  **`thin_grating_efficiency_1d`** instead -- same signature, same
  numerical output.  The 4.0.1 alias is now the canonical name.

### Packaging

* **`validation/` and `tests/` now ship in the source distribution.**
  Added `MANIFEST.in` so that ``pip download lumenairy --no-binary
  :all: && tar xzf ... && python validation/run_all.py`` actually
  works.  Wheels still ship only the library packages.

### GUI internals

* `distortion_dock`, `footprint_dock`, `spot_field_dock`,
  `sensitivity_dock` now delegate to the public analysis functions.
  Rendering is unchanged; the bodies are roughly half their previous
  size.

### Validation

* 1 new validation file: `validation/analysis/test_field.py`
  (22 tests; all pass).
* All 30 pre-existing validation files still pass after the rename
  + refactor.
* Existing `test_thin_grating_alias_matches_rcwa_1d` regression test
  replaced with `t_thin_grating_energy_conserved_long_period`
  (energy conservation on a long-period grating).

## [4.3.0] — 2026-05-13

**Diffractive optics + off-axis Seidel + module organization + unit-test
layer.**  Pure-additive; no breaking changes.

### Added — Diffractive lens trio

* **`create_diffractive_lens(N, dx_m, focal_length_m, wavelength_m, *,
  center=(0,0))`** -- continuous-phase thin-lens equivalent
  ``exp(-i k r^2 / (2 f))``.  Reference implementation; the limiting
  case of `create_kinoform` as `n_levels -> inf`.
* **`create_kinoform(N, dx_m, focal_length_m, wavelength_m, *,
  n_levels=8, center=(0,0))`** -- multi-level quantized phase
  diffractive lens.  Diffraction efficiency
  ``eta_1 = sinc^2(1/n_levels)``: ~40.5% at 2 levels, ~81% at 4,
  ~95% at 8, ~99% at 16.
* **`create_fresnel_zone_plate(N, dx_m, focal_length_m, wavelength_m,
  *, binary=True, n_zones=None, center=(0,0))`** -- classical
  amplitude FZP (``binary=True``, default; ~10% efficiency) or
  Rayleigh-Wood phase FZP (``binary=False``; ~40% efficiency).
  ``n_zones`` crops to a finite aperture.

All three exported at top-level (`lumenairy.create_diffractive_lens`,
etc.) and from `lumenairy.elements`.  16 new validation tests in
`validation/elements/test_doe.py`.

### Added — Off-axis Seidel analysis

New module `lumenairy.raytrace.seidel_analysis`:

* **`seidel_field_sweep(surfaces, wavelength, field_heights, *,
  object_distance=inf, stop_index=None)`** -- evaluates Seidel
  Hopkins sums at a grid of field heights in one call.  Returns
  per-surface arrays of shape ``(N_surfaces, N_fields)`` and total
  sums of shape ``(N_fields,)``, suitable for plotting S1-S5 vs
  field angle, finding zero-coma stops, etc.
* **`seidel_wfe(seidel_or_totals, rho, theta, *, field_index=None,
  field_angle=None)`** -- reconstructs ``W(rho, theta)`` from a
  Seidel total dict using the standard Welford expansion (the
  Petzval S4 term is scaled by ``sigma^2`` automatically using the
  field-angle metadata that `seidel_coefficients` now exposes).

`seidel_coefficients` is unchanged externally except that the
returned dict now also contains ``'field_angle': float`` (additive
key, backward-compatible).  Existing callers see no behavior change.

10 new validation tests in `validation/raytrace/test_seidel_field.py`
verify the scaling laws (S1/S4 field-independent, S2 ~ h, S3 ~ h^2)
and the WFE reconstruction.

### Added — Module organization polish

Two functions moved to their conceptual home with full
backward-compat re-exports:

* `lumenairy.ao` -> `lumenairy.analysis.ao` (AO is analysis /
  control, not a top-level element family).  `from lumenairy.ao
  import DeformableMirror` still works via a shim.
* `coronagraph_contrast_curve` moved from
  `lumenairy.elements.elements` to a new
  `lumenairy.analysis.coronagraph` (it's analysis, not an element
  factory).  `from lumenairy.elements import
  coronagraph_contrast_curve` still works via a deferred-import
  shim.
* New `lumenairy.elements.coronagraph` namespace module re-exports
  the four coronagraph builders (`apply_lyot_focal_plane_mask`,
  `apply_vortex_phase_mask`, `apply_lyot_stop`,
  `apply_apodized_pupil`) for discoverability.

### Added — Unit-test layer

New `tests/unit/` directory with five focused modules:

* `test_elements_lens.py` -- thin-lens and diffractive-lens
  contracts (dtype, sign convention, phase-only unitarity,
  n_levels quantization).
* `test_propagation.py` -- ASM round-trip, Fresnel shape, resample
  contracts.
* `test_sources.py` -- shape / dtype / centroid / `(E, x, y)` tuple
  return convention for every source factory.
* `test_analysis.py` -- Strehl-of-self = 1, beam power scales as
  |E|^2, Zernike decompose API.
* `test_raytrace.py` -- ABCD det = 1, paraxial helpers.

52 tests total, run in under 1 second.  Run with
``pytest tests/unit``.  The existing pytest wrapper around the
full validation suite moved to
`tests/integration/test_validation_files.py` (run with
``pytest tests/integration``).

### Changed (backward-compat)

* `seidel_coefficients` result dict now includes the key
  ``'field_angle'``.  Existing code that accessed `S1`, `S2`, ...,
  `total`, `labels`, etc., is unaffected.

### Validation

* All 29 pre-existing validation files pass.
* 2 new validation files (`test_doe.py`, `test_seidel_field.py`)
  pass 26/26.
* 52 unit tests pass in 0.76s on a fresh checkout.
* Total: **31 validation files, 26 new validation tests, 52 unit
  tests -- all green.**

## [4.2.0] — 2026-05-12

**Cross-library pupil-grid factories.**  Pure-additive; no breaking
changes.

### Added

* **`zemax_pupil_grid(N=512, clip_to_disk=True)`** -- returns the
  deterministic ``(px, py)`` pupil-sample coordinates used by
  Zemax OpticStudio's ``WavefrontMap`` analysis at sampling N
  (power-of-two: 32/64/128/256/512/1024/2048).  Spacing is
  ``2 / (N - 1)``; both endpoints included.  Pair with
  ``eval_image_plane_wfe(..., pupil_grid=...)`` for a true
  per-ray Lumenairy-vs-Zemax comparison with zero KDTree-NN
  interpolation noise.

* **`chebyshev_pupil_grid(N=31, clip_to_disk=True)`** --
  Chebyshev-Gauss-Lobatto nodes (cluster at the rim).  Matches
  OPDPy's ``OPDSystem.sample()`` grid; ideal for high-order
  polynomial WFE fits and as the "old methodology" path in the
  cross-check repo.

Both factories return plain NumPy arrays so users can freely
filter, transform, or compose them.  Top-level exports
``la.zemax_pupil_grid`` and ``la.chebyshev_pupil_grid``.

### Validation

* 6 new tests in ``validation/analysis/test_image_plane_wfe.py``
  verify: Zemax-grid layout formula, clip-to-disk pi/4 ratio,
  full-square mode, integration round-trip with
  ``eval_image_plane_wfe``, Chebyshev rim-clustering, error on
  N <= 0.
* 34/34 tests in the analysis-WFE suite pass.

### Use case

```python
# Match Zemax's WavefrontMap exactly at 512x512:
px, py = la.zemax_pupil_grid(N=512)
wfe = la.eval_image_plane_wfe(prescription, wavelength=587.56e-9,
                                pupil_grid=(px, py))
# wfe.opd_w[i] is now directly comparable to Zemax WavefrontMap[i]
# with zero KDTree-NN interpolation noise.
```

## [4.1.0] — 2026-05-12

**`pupil_grid` kwarg on `eval_image_plane_wfe` for cross-library
grid matching.**  Pure-additive; no breaking changes.

### Added

* **`eval_image_plane_wfe(..., pupil_grid=(px, py))`** -- the
  function now accepts an arbitrary pair of normalised pupil-
  coordinate arrays (px, py in [-1, 1]) instead of generating its
  own internal square grid.  When given, one ray is launched per
  ``(px[i], py[i])`` pair; coordinates outside the unit disk are
  silently dropped.

  Use cases:
  * **Cross-library validation.**  Evaluate Lumenairy WFE at
    exactly the same pupil sample points rayoptics / Optiland /
    Zemax used, eliminating the nearest-neighbour interpolation
    noise that otherwise dominated the raw cross-library diff
    floor at ~5-10% of WFE RMS.
  * **Chebyshev / Gauss-Lobatto quadrature** for high-order
    polynomial fits where the user wants edge-clustered nodes.
  * **Sparse / structured grids** (rings, fans, custom sample
    layouts) for fast aberration screens.

  The OPDPy / Lumenairy cross-check repo's new
  ``xcheck_grid_matching.py`` script demonstrates the workflow:
  for the L1 plano-convex test lens, switching from the historic
  KDTree-NN comparison to exact grid matching drops the
  Lumenairy-vs-rayoptics raw RMS diff from 50.6 mw to 0.19 mw
  (260x improvement), confirming that the bulk of the prior
  raw-diff floor was geometric grid-layout noise rather than
  real physics-level disagreement.

### Validation

* 4 new tests in
  ``validation/analysis/test_image_plane_wfe.py`` cover the
  ``pupil_grid`` kwarg: 8-ray ring is traced exactly, out-of-disk
  coords are silently dropped, empty / all-outside grids raise
  ``ValueError``, and a Chebyshev grid produces finite WFE.
* Full suite: 29/29 files pass.

## [4.0.1] — 2026-05-12

**Bug-fix patch.**  A deep multi-agent audit of the 4.0.0 codebase
surfaced five real / latent bugs (plus several non-bug findings
that were verified-then-dismissed).  This release ships fixes
plus regression tests for each.  Pure-additive; no breaking
changes.

### Fixed

* **`eval_image_plane_wfe` chief-ray `N=0` fallback was wrong physics.**
  The new off-axis path I shipped in 4.0 had a fallback
  ``N_chief = float(Nd[chief]) if Nd[chief] != 0 else 1.0`` that
  used an exact-zero comparison and a non-physical unit-vector
  default for grazing-incidence chief rays.  Replaced with
  ``abs(N_chief) < 1e-12`` -> returns ``(nan, nan)`` so the failure
  is visible rather than silent.  See
  `lumenairy/analysis/image_plane_wfe.py`.

* **Ray-trace direction-cosine renormalisation silently promoted
  zero-magnitude rays to bogus unit vectors.**  After `_refract` and
  `_reflect`, the renormalisation step was
  ``mag = np.maximum(mag, 1e-30); rays.L /= mag``, which turned
  a numerically-degenerate ``(0, 0, 0)`` direction-cosine vector
  into a unit vector along ``(0, 0, 1e-30)``.  Now flags such rays
  with ``RAY_NAN`` so downstream diagnostics catch the pathology.
  See `lumenairy/raytrace/core.py` lines 615-664.

* **`BSDFModel` base class was a regular class with `NotImplementedError`
  in its methods** -- meaning users who accidentally instantiated
  the base got a deferred runtime error at first method call
  instead of a loud failure at construction.  Promoted to an
  explicit `abc.ABC` with `@abstractmethod` decorators on
  `evaluate` and `sample`.  Direct instantiation now raises
  `TypeError` immediately.  Concrete subclasses (`LambertianBSDF`,
  `GaussianBSDF`, `HarveyShackBSDF`) unchanged.

### Documentation

* **`rcwa_1d` was misadvertised.**  The function name implies full
  Rigorous Coupled-Wave Analysis but the implementation is an
  analytical thin-grating scalar approximation (reflection
  hardcoded to zero, no S-matrix interface matching).  Added the
  honest-name alias `thin_grating_efficiency_1d` and reserved
  `rcwa_1d` for a future genuine RCWA implementation.  Existing
  callers continue to work; new code should prefer the honest
  name.

* **`create_hermite_gauss` / `create_laguerre_gauss` normalisation
  inconsistency with the asymptotic modal propagator.**  The
  source helpers use grid-numerical power-normalisation while the
  asymptotic-propagator modal basis uses analytical
  normalisation.  These agree to ~1e-6 on typical grids
  (``L >= 4 w0``) but can differ on tight grids that clip the
  Gaussian tails.  Added a `Notes` section to both function
  docstrings calling this out, with guidance to prefer the
  asymptotic-module's analytical normalisation when chaining
  through the modal-asymptotic propagator.

### Validation

* 4 new regression tests guard against each of the verified bugs
  re-emerging:
  - `validation/analysis/test_image_plane_wfe.py`: chief-ray
    N-fallback uses the new `abs() < eps` guard + NaN return.
  - `validation/raytrace/test_raytrace.py`: `_refract` / `_reflect`
    flag zero-mag rays as `RAY_NAN`; normal refraction still
    returns all alive rays.
  - `validation/raytrace/test_raytrace.py`: `thin_grating_efficiency_1d`
    bit-identical to `rcwa_1d`.
  - `validation/analysis/test_features.py`: `BSDFModel()` direct
    instantiation raises `TypeError`; concrete subclasses still
    instantiable.
* Full validation suite: 29/29 files pass.

## [4.0.0] — 2026-05-12

**Polish + Tier-1-gap-closing release.**  A four-tier audit (API
consistency, sign / unit conventions, cross-module pipelines, and
peer-library feature parity) surfaced a set of verified bugs and
genuine functional gaps; 4.0 ships fixes for all of them plus a
batch of new helpers that compose with the existing infrastructure.
Pure-additive with one soft-breaking convention fix (see below).

### Added — Adaptive optics primitives

* **`lumenairy.ao`** -- new module exposing closed-loop AO building
  blocks:

  * `DeformableMirror(n_actuators, pitch, dx, N,
    inter_actuator_coupling=0.15)` -- Gaussian-influence-function DM
    on a Cartesian actuator grid.
  * `zernike_modal_basis(n_modes, n_lenslets, semi_aperture,
    first_mode=1)` -- builds the slope-to-modal reconstruction
    matrix for an OSA-indexed Zernike basis on an SH-WFS lenslet
    grid.
  * `slope_to_modal(slopes, basis)` -- reconstructs modal
    coefficients from SH slopes.
  * `LeakyIntegrator(gain, n_modes, leak=0.0)` -- first-order
    leaky-integrator control law.

  These primitives compose with the existing `shack_hartmann` and
  `generate_turbulence_screen` into a single-conjugate AO closed
  loop.  See the new wiki page **[Function Reference — Adaptive
  Optics](https://github.com/travaj24/LumenAiry/wiki/Function-Reference-Adaptive-Optics)**
  for the end-to-end pipeline.

### Added — Off-axis image-plane wavefront error

* **`eval_image_plane_wfe(field=(Hx, Hy), field_max_m=...)`** -- off-
  axis field points now supported.  Previously raised
  `NotImplementedError` for any `field != (0, 0)`.  The reference-
  sphere kernel now generalises to arbitrary `(cx, cy, cz)` sphere
  centres; the chief ray's transverse position at the image plane
  is computed by free-space propagation from the last lens vertex.
* **`field_grid_wfe(prescription, wavelength, field_max_m, n_field=5,
  ...)`** -- wraps `eval_image_plane_wfe` over an `n_field × n_field`
  grid of normalised field coordinates and returns
  `(Hx, Hy, pv_waves, rms_waves, strehl, img_d_m, wfe_per_field)`
  for the standard "WFE across the field" plot.

### Added — Coronagraph contrast-curve helper

* **`coronagraph_contrast_curve(psf_coro, psf_ref, dx_focal,
  wavelength, f_eff, ...)`** -- radial contrast vs angular separation
  in units of `λ·f/D`.  Three azimuthal-reduction modes
  (`'mean'` / `'median'` / `'rms'`).

### Added — Paraxial-design helpers

* New `lumenairy.raytrace.paraxial` submodule:

  * `field_of_view(prescription, wavelength)` -- `(half_FoV_rad,
    half_FoV_object_m)`.
  * `f_number(prescription, wavelength)` -- paraxial `EFL / D`.
  * `optical_invariant(efl, f_number, ...)` -- Lagrange invariant.
  * `defocus_waves_to_zernike(d)` -- convert geometric defocus to
    the OSA `c(2, 0)` coefficient.
  * `astigmatism_waves_to_zernike(a)` -- same for `(2, ±2)`.

### Added — Prescription schema unification

* **`normalize_prescription(prescription)`** -- returns a deep-
  copied prescription with the canonical superset of schema keys
  populated, regardless of which loader / builder produced the
  input.  Avoids the silent-fallback bugs where (for instance)
  `monte_carlo_tolerancing` only perturbed `'surfaces'` and
  silently skipped mirrors in `'elements'`.

### Added — JonesField bound analysis methods

* `JonesField.stokes_parameters()`, `.degree_of_polarization()`,
  and `.polarization_ellipse()` -- bound forms of the existing
  module-level helpers so chained pipelines
  (`jf.propagate(...).stokes_parameters()`) work without round-
  tripping through the scalar API.

### Added — Storage round-trip dtype preservation

* `save_field_h5(..., preserve_dtype=True)` and
  `save_planes_h5(..., preserve_dtype=True)` -- keep the native
  complex precision (`complex64` / `complex128`) through the file
  round-trip.  Defaults to the historical complex128-coercion
  behaviour for backward compatibility.

### Added — Detector noise realism

* `apply_detector` gains four new kwargs:

  * `hot_pixel_map` -- boolean map of detector defects to saturate.
  * `cosmic_ray_rate` -- Poisson-distributed strike count per
    exposure, uniformly located.
  * `cosmic_ray_amp_e` -- charge per strike [electrons].
  * `bayer_pattern` (`'RGGB'` / `'BGGR'` / `'GRBG'` / `'GBRG'`) +
    `bayer_qe=(R, G, B)` -- per-cell QE mosaic for RGB
    colour-filter arrays.

### Changed — Source normalisation conventions

* `create_gaussian_beam`, `create_hermite_gauss`, and
  `create_laguerre_gauss` all now accept a `normalize` kwarg with
  options `'peak'`, `'power'`, or `'none'`.  Each function preserves
  its historical default (`create_gaussian_beam`: `'peak'`;
  `create_hermite_gauss` / `create_laguerre_gauss`: `'power'`) so
  no existing code breaks.  Pass `normalize='power'` to homogenise
  across the source family.

### Changed — Analysis-layer GPU / JAX preservation

* `beam_centroid`, `beam_d4sigma`, `beam_power`, `strehl_ratio`,
  `compute_psf`, `compute_otf`, and `compute_mtf` now dispatch
  through `lumenairy.backend.array_namespace`.  CuPy / JAX inputs
  are no longer silently coerced to NumPy.

### Changed — Element-family GPU / JAX preservation

* `apply_aperture` and `apply_gaussian_aperture` now dispatch
  through `array_namespace`.

### Fixed — `dy` support on aberration / source helpers

* `apply_zernike_aberration` now accepts a `dy` kwarg (silent
  square-grid assumption removed for rectangular grids).
* `create_hermite_gauss` and `create_laguerre_gauss` now accept a
  `dy` kwarg and an `(Ny, Nx)` tuple for `N`, allowing rectangular
  grids without stretching the mode envelope along y.

### Documentation

* New wiki pages: **Function Reference — Adaptive Optics**,
  **Migration Guide** (POPPy / HCIPy / prysm / Optiland / rayoptics
  → LumenAiry mapping), **Glossary** (sign/unit conventions + ASM
  vs MFT vs Fresnel vs RS decision tree).
* Expanded **Home → "What's in the box"** with high-contrast,
  AO, broadband-imaging, and field-dependent-analysis sections.

### Validation

* 29/29 validation files pass (up from 27/27 in 3.9.0; two new
  files added: `test_ao.py` and `test_coherence.py`).
* Coherence helpers (`koehler_image`, `extended_source_image`,
  `mutual_coherence`) -- previously untested -- now have dedicated
  validation coverage.
* AO primitives: 12 tests covering DM phase-application
  invariants, modal reconstruction round-trips, integrator steady-
  state convergence, and a zero-aberration closed-loop sanity
  check.
* Off-axis WFE: 5 new tests including the `field_grid_wfe` shape +
  on-axis-minimum invariant.
* Paraxial helpers, schema normalisation, contrast curve,
  detector noise, JonesField bound methods, and `dy` support
  all have dedicated tests.

## [3.9.0] — 2026-05-12

Feature release.  Lights up the high-contrast-imaging and
broadband-detector slice of the API surface.  Pure-additive; no
breaking changes.

### Added

* **Coronagraph templates** (`lumenairy/elements/elements.py`):

  * `apply_lyot_focal_plane_mask(E, dx, mask_diameter, profile=...)`
    -- classical Lyot focal-plane occulter.  Profile options
    `'hard'` (binary), `'gaussian'` (smooth-edged), `'sin2'`
    (band-limited).
  * `apply_vortex_phase_mask(E, dx, charge=2)` -- scalar focal-plane
    vortex applying `exp(i * l * theta)`.  Charges 2 (AGPM), 4, 6,
    8 are the standard astronomical values.
  * `apply_lyot_stop(E, dx, outer_diameter, inner_diameter=0)` --
    downstream pupil annulus.  Coronagraph-literature naming
    wrapping the equivalent annular `apply_aperture` call.
  * `apply_apodized_pupil(E, dx, diameter, apodization='cos2')` --
    entrance-pupil graded-transmission apodiser with `'cos2'`,
    `'cos_power'`, `'gaussian'`, and `'sonine'` profile options.

* **Polychromatic PSF accumulator** (`lumenairy/analysis/analysis.py`):

  * `polychromatic_psf(prescription, wavelengths, weights, N, dx,
    *, E_in=None, image_distance=None, normalize='power',
    bandlimit=True, return_components=False)` -- returns the full
    integrated intensity map on a common image plane plus
    diagnostic metrics (centroid wavelength, per-lambda peak +
    Strehl, accumulated centroid, D4-sigma).  Complements the
    existing `polychromatic_strehl`, which only returns scalar
    Strehl ratios.

### Validation

* 7 new tests in `validation/elements/test_elements.py`: hard /
  Gaussian Lyot mask profiles, charge-2 vortex phase topology,
  Lyot stop annulus, cos^2 apodisation monotonicity, Sonine
  exponent throughput ordering, and an end-to-end vortex+Lyot
  pipeline verifying >100x on-axis starlight suppression
  (charge-2 vortex + 0.85*D Lyot stop).
* 2 new turbulence tests: Kolmogorov 5/3 structure-function
  log-log slope; finite outer-scale L0 reduces phase variance
  (von Karman branch).
* 6 new tests in `validation/analysis/test_analysis.py` covering
  `polychromatic_psf` power / peak normalisation, on-axis
  centroid, per-wavelength component sum, chromatic broadening,
  and centroid wavelength.
* Full validation suite (27/27 files) passes.

### Documentation

* New wiki page: **Function Reference - Coronagraphs** with full
  parameter tables, end-to-end pipeline walkthrough, and
  references (Lyot 1939, Mawet 2005, Soummer 2005/2007, Kasdin
  2003).
* **Function Reference - Propagation** gains 154 lines documenting
  the three matrix-Fourier-transform propagators
  (`fresnel_propagate_mft`, `fraunhofer_propagate_mft`,
  `angular_spectrum_propagate_mft`) that shipped in 3.5.7 but
  were thinly documented; arbitrary-output-grid (MFT family)
  overview + use-case breakdown + examples.
* **Function Reference - Analysis** gains a `polychromatic_psf`
  section with a "when to use which" callout vs
  `polychromatic_strehl`.
* **Home**, **Function Reference**, and **Sidebar** updated with
  cross-links and new high-contrast / broadband-imaging bullets.

## [3.8.3] — 2026-05-11

Documentation patch.  README.md leads with the 3.8.2 release notes
(it had been left at the 3.8.0 header through the 3.8.1 and 3.8.2
cuts, so the PyPI project description still showed the 3.8.0
"What's new" section for users running ``pip show lumenairy``).
No code change; bumping the patch version so that the next PyPI
publish picks up the corrected README.

## [3.8.2] — 2026-05-11

Adds two optional convention controls to
`lumenairy.eval_image_plane_wfe` so the function can be aligned
exactly with Zemax OpticStudio / rayoptics / Optiland defaults
when doing cross-library validation, and so it can also report
"best-focus" wavefront error (the focus a lab tech finds by
maximising spot intensity, which is what published Strehl ratios
implicitly assume).

### Added

* **`image_plane` parameter on `eval_image_plane_wfe`**, default
  `'paraxial'` (back-compat).  Accepts:

  * `'paraxial'` -- chief paraxial focus from the Gauss imaging
    equation + principal planes.  Matches Zemax `WavefrontMap`
    default, rayoptics `foc=0`, Optiland default.  Use for
    cross-library validation.
  * `'best_rms'` -- closed-form shift to the image plane that
    minimises WFE RMS.  Fits a defocus coefficient to the
    paraxial-focus WFE and converts via
    `1/R' = 1/R + 2*c1*lambda/r_pupil^2`.  No iteration.
  * `'best_pv'` -- 1-D numerical minimisation of PV via
    `scipy.optimize.minimize_scalar` (with a 21-point coarse-
    scan fallback if scipy is unavailable).  Useful for
    PV-defined tolerance specs.
* **`sphere_tangent` parameter on `eval_image_plane_wfe`**,
  default `'vertex'` (back-compat).  Accepts:

  * `'vertex'` -- reference sphere tangent at the LAST LENS
    SURFACE vertex; radius = `img_d_m`.  Simplest convention;
    matches what the 3.8.0 / 3.8.1 versions of this function
    used.
  * `'exit_pupil'` -- tangent at the exit pupil; radius =
    `img_d_m - xp_z` (using the signed exit-pupil offset from
    `first_order_data`).  Matches rayoptics / Optiland / Zemax
    internal conventions.  Mathematically a no-op for the WFE
    of chief-centred spheres (chief sits on every concentric
    sphere with zero OPD), but adopting this convention makes
    the reported sphere radius match other libraries' reported
    values for downstream interoperability.
* **`ImagePlaneWFE` dataclass gained four new fields**:
  `image_plane`, `sphere_tangent`, `r_sphere_m` (signed
  reference-sphere radius actually used), and
  `img_d_m_paraxial` (the pre-shift paraxial image distance, so
  the longitudinal shift `img_d_m - img_d_m_paraxial` is easy to
  read out for `'best_rms'` / `'best_pv'` modes).

### Validation

* `validation/analysis/test_image_plane_wfe.py` extended from 11
  to 18 checks: default-equals-paraxial back-compat,
  best_rms-reduces-RMS, best_pv-reduces-PV, image-plane shift is
  recorded, sphere-tangent rebases radius, sphere-tangent
  vertex-vs-exit_pupil WFEs agree within FP noise (verifies the
  no-op claim from CROSS_CHECK_METHODOLOGY.md), and invalid
  argument raises `ValueError`.  All 18 pass.

### Backwards compatibility

Fully back-compatible.  Default values of both new parameters
(`image_plane='paraxial'`, `sphere_tangent='vertex'`) reproduce
3.8.0 / 3.8.1 output bit-for-bit.  No changes to
`apply_real_lens_traced`, the `trace()` ray-trace primitive,
`first_order_data`, or `remove_low_order_aberrations`.

## [3.8.1] — 2026-05-11

Patch release.  Single fix: `lumenairy.__version__` was inadvertently
left at `"3.7.10"` in the 3.8.0 source even though `pyproject.toml`
bumped to `3.8.0` (so PyPI metadata and editable installs reported
mismatched versions).  This release re-syncs the two.  No API
changes.

## [3.8.0] — 2026-05-11

Image-plane wavefront-analysis release.  Adds three peer-library
features that have been missing from the public API up to now:
direct image-plane reference-sphere OPD evaluation, a unified
first-order paraxial summary, and a best-fit low-order aberration
removal utility.  All three were validated by a 13-lens
cross-check against rayoptics + Optiland + OPDPy (see the
`OPDPy_Lumenairy_Crosscheck/` companion repo), which is also the
context in which the API was designed.

### Added

* **`lumenairy.analysis.eval_image_plane_wfe(prescription,
  wavelength, field=(0,0), n_pupil=31, img_d_m=None)`** — direct
  image-plane reference-sphere wavefront-error evaluation.
  Returns an `ImagePlaneWFE` dataclass with per-ray pupil
  coordinates, OPD in waves, alive mask, and convenience
  properties `pv_waves`, `rms_waves`, `strehl` (Marechal
  approximation).  Uses an EXACT ray-sphere intersection from
  the actual lens-exit ray state `(x, y, z, L, M, N)` to the
  reference sphere centered on the chief ray's image-plane
  intersect, so it's accurate for fast f/2 systems and
  diverging singlets (negative `img_d_m` supported).  This is
  the standard textbook Zemax / Code V WFE convention, and is
  the natural complement to `apply_real_lens_traced` which
  returns lens-exit chief-relative OPL (the right input for
  downstream ASM / Fresnel propagation).
* **`lumenairy.remove_low_order_aberrations(opd_w, px, py,
  include_r4=True)`** — best-fit subtraction of piston + tilt +
  defocus + 4th-order spherical from a scattered OPD field.
  The residual is the genuinely-higher-order aberration content
  (6th-order SA, coma, astigmatism, ...) where independent
  ray-trace implementations actually diverge — the realistic
  apples-to-apples cross-library comparison metric.  See the
  cross-check methodology document for why best-fit removal is
  preferable to "common-sphere projection" between libraries
  that all use chief-ray-centered reference spheres of
  different radii.
* **`lumenairy.raytrace.first_order_data(surfaces_or_prescription,
  wavelength, stop_index=None)`** — comprehensive paraxial
  first-order summary in a single call.  Returns
  `FirstOrderData` with EFL, BFL, FFL, EP/XP positions and
  radii, principal-plane offsets, working f-number, the full
  ABCD matrix, and stop index.  Includes a `.summary(units)`
  convenience formatter for printout.  Combines
  `system_abcd`, `compute_pupils`, and the standard
  focal-length / principal-plane geometry into one record;
  used internally by `eval_image_plane_wfe` to derive the
  finite-conjugate image distance from the Gauss imaging
  equation + principal-plane offsets.
* **`lumenairy.ImagePlaneWFE`**, **`lumenairy.FirstOrderData`** —
  dataclass result types for the above, also re-exported from
  the top-level `lumenairy` namespace.

### GUI

* `SystemSummaryWidget` (Analysis tab "Summary" dock) now
  appends an **Image-plane WFE (on-axis)** block reporting
  PV / RMS / Marechal Strehl and the after-best-fit residual
  RMS, computed live from the current prescription.  No new
  controls; the block appears automatically when the system
  has a valid finite-conjugate object distance and prescription.
  See `GUI_CHANGELOG.md` for the per-tab dock placement.

### Validation

* New `validation/analysis/test_image_plane_wfe.py` — 11 checks
  spanning `first_order_data` paraxial geometry (EFL, H/H' for
  plano-convex and thick equiconvex, f-number), chief-at-zero +
  sign convention + PV agreement with rayoptics (5% tolerance)
  + aplanatic-conic-reduces-RMS for `eval_image_plane_wfe`, and
  pure-defocus / r⁴-toggle / orthogonal-astigmatism-preserved
  checks for `remove_low_order_aberrations`.  All 11 pass.
* The 27-file validation suite (`validation/run_all.py`) was
  unchanged; the new file brings the total to 27.

### Notes

* `find_paraxial_focus(surfaces, wavelength)` still returns the
  back focal length (infinity-conjugate focus).  For
  finite-conjugate imaging, `eval_image_plane_wfe` now computes
  the correct image distance internally from the Gauss equation +
  principal-plane offsets.  A standalone
  `find_image_distance(surfaces, wavelength, object_distance)`
  helper may be added in a later release; for now use
  `1 / (1/efl - 1/(obj_d + pp_object_z)) + pp_image_z` from a
  `first_order_data` call.

### Backwards compatibility

No removals.  No changes to `apply_real_lens_traced`, the
ray-trace `trace()` primitive, or any pre-3.8 public API.
The three new functions are purely additive.  All 27
validation files pass.

## [3.7.10] — 2026-05-11

GUI-only release.  No core library API changes; see
`GUI_CHANGELOG.md` for the workspace reorganisation details.

### Backwards compatibility

No public API removals.  All 26 validation suite files pass.

## [3.7.9] — 2026-05-11

GUI-driven quality-of-life release.  Only one core API change:
`SystemModel.trace_started` signal added.  All other work is in
the GUI; see `GUI_CHANGELOG.md` for the user-facing list.

### Added

* `SystemModel.trace_started` — emitted at the top of `run_trace`.
  Lets the GUI raise a busy cursor / status label as soon as the
  trace begins, paired with the existing `trace_ready` for the
  post-trace cleanup.  No-op for non-GUI consumers that haven't
  connected to the new signal.

### Notes

* The default Lumenairy storage path now embeds the active
  prescription (JSON) plus propagator settings inside saved
  HDF5 / Zarr outputs (set via the wave-optics dock; the
  `lumenairy.io.storage.write_sim_metadata` API is unchanged).
  Older files load fine with the new dock's "Load saved run…"
  flow -- the embedded-prescription block is treated as
  optional.

### Backwards compatibility

No removals.  All 26 validation suite files pass.

## [3.7.8] — 2026-05-11

Closes the remaining ray-trace gaps from the 3.7.6 / 3.7.7
world-frame rollout: the paraxial fan helpers
(`ray_fan_data`, `opd_fan_data`) now have world-frame
counterparts, so the Ray Fan dock's tangential / sagittal /
OPD plots are fold-accurate on the same prescriptions where
`run_trace` already was.

### Added

* `ray_fan_data_world(surfaces, wavelength, semi_aperture,
  field_angle, n_rays)` -- identical signature and return
  shape to `ray_fan_data`, but expects each Surface to have
  `world_origin` / `world_R` populated and routes through
  `trace_world` so the per-pupil-coordinate fan rays and
  reference chief ray all land at the correct world image
  position in folded designs.
* `opd_fan_data_world(...)` -- world-frame OPD residuals.
* Both helpers re-exported from `lumenairy.raytrace` for
  drop-in use by the GUI's Ray Fan dock and external scripts.

### Notes

* `ray_fan_plot` (the matplotlib helper that wraps
  `ray_fan_data`) is unchanged -- it remains a paraxial-trace
  shortcut for axial systems.  Use `trace_world` +
  `ray_fan_data_world` for folded systems.

### Backwards compatibility

No public API removals.  All 26 validation suite files pass.

## [3.7.7] — 2026-05-11

Public API for the world-frame trace + GUI-side correctness rollout.
The 3.7.6 release added the world-frame trace path (`trace_world`)
but only the GUI's main `run_trace` used it; analysis docks
(footprint, distortion, spot-vs-field, rayfan field-curvature)
still built their own surface lists via the legacy local-frame
path and inherited the residual ~1/cos(θ) geometry error in
folded designs.  3.7.7 promotes the world-frame surface builder
to a public method and routes those docks through it.

### Added

* `SystemModel.build_trace_surfaces_world()` — public cached
  accessor for the world-frame surface list (one Surface per
  actual optical surface, each carrying absolute
  `world_origin` and `world_R`).  Pair with `trace_world` for
  folded-system-accurate analysis from any consumer that
  currently calls `build_trace_surfaces()` + `trace()`.
* `SystemModel.build_run_trace_world_surfaces(image_distance=
  None)` — world-frame surface list with an image-plane
  Surface appended at the Detector's world frame (or, if no
  Detector / zero distance, at the last optical surface
  advanced by `image_distance` (or the paraxial BFL) along its
  local `+z` axis).  This is the surface list every analysis
  dock needs when it wants "full trace with image plane".

### Changed

* `SystemModel.__init__` and `_invalidate` add
  `self._flat_surfaces_world_cache = None` so the world
  surface list is cached and invalidated on the same
  granularity as the legacy `build_trace_surfaces()` cache.

### Backwards compatibility

No public API removals.  `build_trace_surfaces()` /
`_build_trace_surfaces_internal()` continue to return the
legacy local-frame list (with cb_pre Surfaces) for ABCD /
paraxial helpers and any external consumer that hasn't yet
migrated.  All 25 numerical validation files still pass
unchanged; a new `validation/gui/test_layout_shrink.py`
brings the suite total to 26 files / +14 assertions covering
GUI dock-shrink physics and tx71 world-frame chief-ray
landings.

## [3.7.6] — 2026-05-11

Sequential **world-coordinate** ray trace.  The core trace engine
now has a second sequential trace path (`trace_world`) that
propagates rays in world coordinates between surfaces and only
drops into each surface's local frame for the intersect /
refract / reflect step.  Surfaces in this path carry their own
absolute `world_origin` and `world_R` (local-to-world rotation),
which means folded systems are encoded as "each surface knows
where it actually is in space" rather than as a chain of
coordinate-break frame transforms.  This eliminates the
~`1/cos(θ)` world-distance error that the cb_pre-tilted local
frame introduced for any element following a tilted mirror.

The legacy local-frame `trace()` path is unchanged and remains
the default for code that constructs `Surface` objects directly
(test suites, wave-optics modules, ABCD / paraxial helpers).
The GUI's `SystemModel.run_trace` switches to `trace_world` so
the 2D / 3D layouts, spot diagram, and image-plane trace see
world-accurate ray positions in folded prescriptions.

### Added

* `trace_world(rays, surfaces, wavelength, output_filter,
  surface_diffraction)` in `lumenairy.raytrace.core` — same
  signature as `trace()`, but expects each `Surface` to have
  `world_origin` (m, shape `(3,)`) and `world_R` (shape
  `(3, 3)`).  Re-exported as `lumenairy.raytrace.trace_world`.
* `Surface.world_origin` and `Surface.world_R` optional fields
  (default `None`).  When both are populated, `trace_world`
  treats the surface as positioned at `world_origin` in world
  coords with its local +z axis along `world_R[:, 2]`.  When
  `None`, surfaces are only usable on the legacy `trace()`
  path (no behavioural change for pre-3.7.6 callers).
* `_world_to_local_state(rays, origin, R)` and
  `_local_to_world_state(rays, origin, R)` helpers transform
  a `RayBundle`'s positions and direction cosines between the
  world frame and a given surface-local frame.  Orthogonal
  transforms (preserve unit-length direction vectors).
* `surfaces[i].world_origin` and `world_R` are preserved on
  `Surface` copies (the run_trace fresh-copy loop in
  `SystemModel.run_trace` carries them through).

### Changed

* `SystemModel._build_trace_surfaces_world` emits ONE
  `Surface` per actual optical surface (S1, S2, …) with
  `world_origin` and `world_R` baked in from each
  `Element.origin` / `Element.R` (computed by
  `recompute_element_frames`).  No coord-break Surfaces on
  the world path — tilts and decenters are absorbed into each
  surface's `world_R` and `world_origin`.
* `SystemModel.run_trace` now uses `trace_world` +
  `_build_trace_surfaces_world`.  The legacy
  `_build_trace_surfaces_internal` (with cb_pre Surfaces) and
  the legacy `trace()` are retained as the public
  `build_trace_surfaces()` for ABCD / paraxial / wave-optics
  consumers that don't yet need world-frame accuracy.
* `surface_frames_2d_mm` / `surface_frames_3d_mm` no longer
  emit a separate cb_pre frame for tilted elements — one
  entry per actual surface, matching the world-trace history
  one-to-one.
* `_build_trace_surfaces_internal` now routes the post-mirror
  air gap onto the cb_post Surface's thickness instead of the
  mirror's surface thickness.  This is a band-aid on the
  legacy local-frame path that the world-trace path replaces
  cleanly; both produce the correct world-frame ray
  positions on tx4designstudy71 and similar folded designs.

### Backwards compatibility

No public API removals.  `trace()`, `system_abcd()`,
`paraxial_trace()`, `seidel_coefficients()`, `find_stop()`,
`compute_pupils()`, and `surfaces_from_prescription()` all
unchanged.  `Surface` is a frozen dataclass — the new fields
are optional with `default=None`, so existing constructor
calls and pickled surface lists are unaffected.  All 25
validation suite files pass unchanged.

### Verified on the folded test design

`tx4designstudy71.zmx` (a 1× telecentric design with a
45° fold mirror, six pre-fold lenses + Metasurface + six
return-leg lenses + a 45° fold + SpatialFilter + three more
lenses + Detector): the chief ray's world positions after
`run_trace` now land at each post-fold lens centre to
floating-point precision, with the post-fold world direction
correctly `(0, +1, 0)` matching the post-fold optical axis.

## [3.7.0] — 2026-05-10

Tilt-aware sequential ray tracing.  The core trace engine now
recognises coordinate-break surfaces and transforms the ray
bundle's frame at each one, so folded systems (mirrors with
coord-breaks, decentered lenses) actually deflect rays at
trace time instead of running through the unfolded co-axial
equivalent.  Backward compatible: existing prescriptions
emit no coord-breaks, so the trace path for un-tilted systems
is unchanged and all 25 validation suite files pass without
modification.

### Added

* `Surface` dataclass gains six optional fields: `is_coordbrk`
  (bool), `tilt_x_deg`, `tilt_y_deg`, `tilt_z_deg`,
  `decenter_x_m`, `decenter_y_m`, plus `coordbrk_order` (Zemax
  PARM 6 — 0 = tilts-first-then-decenter, 1 = decenter-first-
  then-tilts).  All default to `is_coordbrk=False`, so existing
  constructor calls and serialised systems are unaffected.
* `_apply_coord_break(rays, surface)` helper transforms a ray
  bundle's local frame in place: subtracts decenter from
  `(x, y)` and rotates position + direction cosines by the
  inverse of the tilt matrix.  Order follows PARM 6.
* `trace()` main loop checks `surf.is_coordbrk` BEFORE
  intersect / refract / reflect and routes to
  `_apply_coord_break`.  The cb is recorded in `ray_history`
  so consumers indexing by surface number stay aligned, and
  its `thickness` carries the ray to the next surface in the
  new (post-transform) frame.
* The GUI's `SystemModel.to_prescription()` now also emits
  `elements` (full chronological list incl. mirrors),
  `all_thicknesses` aligned to it, and `coord_breaks` (one
  entry per tilted element) — the same shape the importer
  produces — so a round-trip through `.zmx` export/import
  preserves fold geometry.

### Changed

* `system_abcd()`, `paraxial_trace()`, and
  `seidel_coefficients()` now skip coord-break surfaces (no
  refractive power) but apply their thicknesses as plain
  transfer matrices so the cumulative axial separation
  matches the trace.  `find_paraxial_focus()` inherits the
  cb-skipping via `system_abcd`.  `find_stop()` and
  `compute_pupils()` need no changes.
* `lumenairy.io.prescriptions.export_zemax_zmx()` dispatches
  to a new cb / mirror-aware writer when the prescription
  carries the new keys.  The new writer emits
  `TYPE COORDBRK` rows with `PARM 1..6`, `GLAS MIRROR` rows,
  and converts physical-positive thicknesses back to Zemax-
  signed (negative post-mirror) by tracking mirror parity.
  Pre-3.7 lens-only prescriptions still go through the
  legacy writer unchanged.

### Backwards compatibility

No public API removals.  `__all__` unchanged.  All 25
validation suite files (`python validation/run_all.py`) pass
unchanged.  Code that constructs `Surface` objects with the
pre-3.7 keyword arguments works identically.

### Known limitations (3.7.x scope)

* Wave-optics propagation (Fresnel / ASM) does not yet honour
  tilt frames; folded systems with wave-level analysis still
  see the un-tilted equivalent.  Plumbing fold-aware Fresnel /
  ASM is queued for 3.8.
* `find_paraxial_focus()` returns the BFL of the un-folded
  paraxial calc; for folded systems the world detector
  position needs a chief-ray chase through the system to find
  the actual focus.  Workaround: place the detector manually
  at the desired distance.
* The optimizer accepts `'tilt_x'` / `'decenter_y'` field
  names in opt-variable tuples, but `set_variable_values`
  writes them to the surface rather than the element-level
  attribute.  Element-level tilt/decenter as variables needs
  a small dispatch fix.
* The JAX trace path mirrors the NumPy trace but does not yet
  implement the cb branch; differentiable trace through
  folded systems falls back to the NumPy path.

## [3.6.1] — 2026-05-10

GUI bug-fix + UX-cleanup release.  No core-library API changes.

* `SystemModel.add_optimization_variable(elem_idx, surf_idx,
  field)` helper added on the GUI-side `SystemModel` (used by
  the new "Attach slider to this parameter…" right-click flow);
  not part of the public `lumenairy` API surface.
* See [GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full GUI-side
  changes: source glyphs in 2D and 3D layouts, bidirectional
  table-layout selection highlighting, immediate refresh on
  source-type change, world-frame ray rendering, 3D camera
  persistence across parameter edits, workspace-defaults trim
  with migration prompt, source-preview rays, OSLO-style attach-
  slider, and the launcher / script-alias cleanup.

### Removed

* Legacy `optical-designer` console script and
  `run_optical_designer.py` launcher (3.5.9 backward-compat
  aliases for the rename).  Use `lumenairy-designer` or
  `python run_lumenairy_designer.py` instead.  QSettings storage
  key unchanged so existing user customisations survive.

### Backwards compatibility

No public API changes.  `__all__` unchanged.

## [3.6.0] — 2026-05-09

GUI feature-coverage release.  No core-library API changes; all
work in this release surfaces previously-unwrapped library
capabilities through the LumenAiry Designer GUI.  See
[GUI_CHANGELOG.md](GUI_CHANGELOG.md) for the full list.

Highlights of GUI-facing changes:

- 5 new specialty docks (Richards-Wolf vector diffraction,
  Köhler partial coherence, Shack-Hartmann sensing, LG aberration
  tensor, RCWA grating analyser).
- Wave Optics dispatch dropdown now also exposes whole-prescription
  GBD / HFPI / Huygens-Fresnel / Subaperture propagators.
- Wave Optics Quick-run preset bar (Fast / Production / Sub-nm).
- Detector model toggle (apply_detector) on the focal-plane field.
- Source factories `Source.top_hat` and `Source.fiber_mode`
  reachable from Insert > Source and the source-row form.
- F6/F7/F8/F9/F10 keyboard shortcuts for retrace / through-focus /
  Zernike / PSF-MTF / caustic.
- Sortable + filterable Keyboard Shortcuts dialog.
- In-app "What's New" modal triggered once per release version.
- Help menu expanded with Wiki, Examples, GUI README, Bug
  tracker, Open Demo, What's New links.
- About dialog now lists detected optional dependencies.
- Tools menu expanded (Scale system, Find nearest Thorlabs,
  Quick Zernikes from trace, Chromatic focal shift).
- Welcome dock redesigned with a hero "Open Demo" button + a
  dedicated "Open Python REPL" button + drag-drop / REPL hints.
- Optimizer dock: primary "Optimize" button + advanced disclosure
  (Global Search, Wave Optimize); JAX checkbox moved into a
  Compute backend group.
- Run buttons standardised across all docks via a single
  `objectName('run_button')` stylesheet rule.
- Status-bar EFL / BFL / f# / EPD / λ metrics are now clickable
  (raise the System Data dock).
- Workspace migration now union-merges new docks into existing
  workspaces so upgraders pick up new specialty docks without
  losing customisations.
- Tolerance dock surfaces its MC limitations up-front and the
  Export Report schema includes a `limitations` field.
- Welcome dock + REPL banner advertise `lumenairy` examples /
  pre-bound `la` namespace for transcribing library examples.

### Backwards compatibility

No core-library API changes.  `__all__` unchanged.  GUI tracks
library version per the lockstep policy in `GUI_README.md`.
QSettings storage key unchanged so existing user customisations
survive the upgrade.

## [3.5.9] — 2026-05-09

GUI catch-up release.  No core-library API changes; this version
exists primarily to surface the 3.5.7-3.5.8 propagator family and
the 3.3.x-3.5.4 analysis utilities through the
**LumenAiry Designer** GUI, which had been pinned to 3.2.14
behaviours.  Library + GUI versions are once again in lockstep
per the policy stated in `GUI_README.md`.

See [`GUI_CHANGELOG.md`](GUI_CHANGELOG.md) for the full GUI-side
changes (Wave Optics dock dispatch, new Caustic Diagnostic dock,
optimizer JAX toggle, Tools menu, Custom-MHS-chain tab, app
rename to LumenAiry Designer).

### Backwards compatibility

No core-library API changes.  `__all__` unchanged.  The Optical
Designer → LumenAiry Designer rename ships the
`run_optical_designer.py` launcher and `optical-designer` script
entry point as backward-compatible aliases for the new
`run_lumenairy_designer.py` / `lumenairy-designer` names.

## [3.5.8] — 2026-05-09

H-cache standardisation across the rest of the propagator family.
Rayleigh-Sommerfeld picks up an H-cache + optional Matsushima
bandlimit + full alignment with the standardisation already used
by `angular_spectrum_propagate` and the MFT propagator family;
the same caching pattern is then ported to
`angular_spectrum_propagate_tilted`,
`scalable_angular_spectrum_propagate`, and
`angular_spectrum_propagate_mft`.

### Added

* **H-cache on `rayleigh_sommerfeld_propagate`.**  The FFT'd Green's
  function ``H = FFT(h)`` is now cached on the NumPy backend keyed on
  the geometry signature ``(2*Ny, 2*Nx, dy, dx, wavelength, z,
  bandlimit, dtype)``.  Repeat calls at the same geometry skip the
  spatial-domain kernel construction and its FFT, which on a NumPy
  CPU backend at lambda=1.31um, dx=2um is roughly:

  | Grid | Cold | Warm | Speedup |
  |---|---|---|---|
  | 512  | ~230 ms  | ~110 ms  | **~2.1x** |
  | 1024 | ~670 ms  | ~270 ms  | **~2.4x** |
  | 2048 | ~1640 ms | ~390 ms  | **~4.2x** |

  The cache is shared with
  `angular_spectrum_propagate` and obeys the same byte budgets
  configured via :func:`set_asm_cache_size`; CuPy device arrays and
  JAX traced arrays continue to rebuild every call (host-side dict
  cannot safely retain device pointers / abstract tracers).  An ``'RS'``
  discriminator string in the key keeps RS entries disjoint from ASM
  entries even when the unpadded grid sizes happen to coincide.

* **`bandlimit=False` kwarg on `rayleigh_sommerfeld_propagate`.**
  Optional Matsushima-style frequency cutoff applied to the FFT'd
  kernel.  Cutoff is computed on the padded (2N x 2N) grid so the
  resulting bandwidth budget matches the FFT length actually used
  by the convolution:

      f_max = (2 * N * dx) / (2 * lambda * |z|)

  Default `False` preserves the historical "exact Green's function"
  character of RS that justifies its use over ASM in the near field.
  Set `True` to suppress aliasing artifacts on coarse grids at long
  propagation distances (the same regime where ASM's `bandlimit=True`
  default is needed).  Reference: Matsushima & Shimobaba 2009,
  Opt. Express 17(22):19662.

* **`'rs'` short alias for `wave_propagator='rayleigh_sommerfeld'`**
  in :func:`apply_real_lens` and :func:`apply_real_lens_traced`.  The
  dispatcher now also forwards the function's ``bandlimit`` kwarg to
  the RS path so users who deliberately enable bandlimit get the
  same treatment in the in-glass propagation step that they get on
  the ASM path.

* **H-cache on `angular_spectrum_propagate_tilted`.**  Same
  ``_h_cache_lookup`` pattern as ASM and RS, keyed on
  ``(Ny, Nx, dy, dx, wavelength, z, fx0, fy0, bandlimit, dtype)`` --
  the tilt-shifted carrier frequencies ``fx0 = sin(tilt_x)/lambda``
  and ``fy0 = sin(tilt_y)/lambda`` fully encode the propagation-
  direction shift, so arbitrary tilt angles are cacheable.  Measured
  ~1.5-1.7x warm-call speedup at N >= 1024.  ``'ASM_TILTED'`` tag
  keeps tilted-ASM entries disjoint from plain-ASM entries.

* **Kernel-bundle cache on `scalable_angular_spectrum_propagate`.**
  The three padded-grid kernels SAS builds per call -- the
  ASM-minus-Fresnel precompensation ``delta_H``, the Fresnel input
  chirp ``H1``, and the optional output-plane quadratic ``H2`` --
  are now cached together as a tuple under one key
  ``(N_new, dx, lambda, z, skip_final_phase, dtype)``.  The internal
  ``_h_cache_store`` accounts for tuple bundles in its byte budget
  via the new ``_entry_bytes`` helper.  Measured ~2x warm-call
  speedup across N=512-2048 (consistent because all three kernels
  contribute substantially at the (pad x N) padded grid).

* **Input-grid H cache on `angular_spectrum_propagate_mft`.**  The
  ASM transfer function on the input frequency grid depends only on
  ``(Ny_in, Nx_in, dy_in, dx_in, wavelength, z, bandlimit, dtype)`` --
  the user-specified output grid (``dx_out``, ``N_out``,
  ``centre_out``) only enters in the Bluestein step downstream.
  Caching the input H lets repeat calls at one input geometry onto
  *different* output grids share one H build.  Measured **~2.7x
  amortised speedup** across multiple ``centre_out`` values at
  N=1024 -- the natural pattern when probing several focal-plane
  zooms from one input field.  ``'ASM_MFT'`` tag keeps these entries
  disjoint from plain-ASM, since the MFT builder uses
  ``|fx| <= fx_max`` (closed boundary) where plain ASM uses
  ``|fx| < fx_max`` (open boundary) -- a one-bin difference at the
  Nyquist frequency that we preserve.

### Standardisation

* `rayleigh_sommerfeld_propagate` now infers the target complex dtype
  from the input field (`E_in.dtype`) and falls back to
  `DEFAULT_COMPLEX_DTYPE` for non-complex input, matching
  `angular_spectrum_propagate`, the MFT propagator family, and the
  rest of the library.  Previously the RS path always upcast to
  complex128 regardless of input precision.

* `angular_spectrum_propagate_tilted` now also infers
  ``target_cdtype`` from input dtype, matching the same convention.

### Internals

* New ``_entry_bytes`` helper in
  ``lumenairy/propagators/propagation.py``: handles both single-array
  H-cache entries (the existing ASM / RS / tilted-ASM / ASM-MFT case)
  and tuple-bundled entries (the new SAS case).
  ``_h_cache_store``'s per-entry size cap and total-bytes eviction
  loop both go through this helper, as does the eviction loop in
  :func:`set_asm_cache_size`.

### Validation

* `validation/propagators/test_propagation.py` adds six tests
  (now **46/46 passing**):
  - RS H-cache idempotency (cached vs fresh output bit-for-bit equal)
  - RS ``bandlimit=True`` accepted and agreeing with ``False`` at
    well-sampled z
  - ``wave_propagator='rs'`` produces the same field as
    ``wave_propagator='rayleigh_sommerfeld'`` through
    :func:`apply_real_lens`
  - Tilted-ASM H-cache idempotency (under ``bandlimit=True``)
  - SAS kernel-bundle cache idempotency (both default and
    ``skip_final_phase=True`` branches)
  - ASM-MFT input-H cache idempotency on identical calls and
    correctness reuse on a second output grid

### Backwards compatibility

No existing API behaviour changes.  `bandlimit` defaults to `False`
on RS, preserving the prior numeric output exactly.  `__all__`
unchanged.

## [3.5.7] — 2026-05-08

Inter-library compatibility improvements: a phase-convention conversion
utility (driven by a multi-library cross-validation against LightPipes,
prysm, POPPy, diffractio, and OPDPy) and three new arbitrary-output-grid
"MFT" propagators that close the focal-zoom capability gap with POPPy
and prysm.

### Added

* **``apply_fresnel_curvature(E, dx, wavelength, R, sign=+1, dy=None)``**
  -- new public utility for round-tripping between phase conventions
  when comparing Lumenairy's complex-field outputs against ray-trace-rooted
  aberration-analysis tools.

  Lumenairy and the Fresnel/ASM-propagator family (LightPipes, prysm,
  POPPy, diffractio, Zemax POP) keep the **absolute physical phase** at
  the output plane: ``arg(E)`` is what a co-propagating plane-wave
  reference (or a coherent receiver) would actually measure.  Other
  excellent tools -- notably **OPDPy** and Zemax wavefront operands
  like ``OPDX`` / ``RWFE`` -- instead store the chief-relative OPD,
  which is purpose-built for aberration analysis (Strehl, RMS WFE,
  Zernike decomposition) and implicitly removes the natural
  Gaussian-beam Fresnel curvature at the image plane that would
  otherwise be a large nuisance term.  The two outputs differ by
  exactly ``exp(i*k*r^2/(2*R))`` with ``R = v - f`` (image distance
  minus focal length) for a thin-lens imager, predictable from
  Gaussian-beam ABCD theory.

  ``apply_fresnel_curvature`` adds (or removes, with ``sign=-1``) this
  curvature so users can convert between the two conventions:

  .. code-block:: python

      # Convert OPDPy / Zemax-OPD output to Lumenairy / LightPipes:
      E_absolute = la.apply_fresnel_curvature(
          E_chief_relative, dx, wavelength, R=v - f)

      # Convert Lumenairy / LightPipes output to chief-relative:
      E_chief_relative = la.apply_fresnel_curvature(
          E_absolute, dx, wavelength, R=v - f, sign=-1)

  Empirical multi-library cross-check (1 mm Gaussian, thin lens
  f=100 mm, 1:1 conjugate, lambda=1310 nm):

  | Library | Convention | Complex correlation vs Lumenairy (post-alignment) |
  |---|---|---|
  | prysm `free_space` | absolute (paraxial Fresnel) | **1.00000** |
  | LightPipes `Forvard` | absolute (exact ASM) | **0.996** (sub-pixel grid offset) |
  | POPPy `propagate_fresnel` | absolute (auto-rescaling Fresnel) | matches qualitatively (R = 113.9 mm fit; predicted 100 mm) |
  | OPDPy AS / Maslov | chief-relative | **0.99996** *after applying R = v - f Fresnel correction* |

  The new wiki page **Phase Conventions and Inter-Library Comparison**
  documents the conventions for each library, when each form is
  appropriate, and how to convert between them.

* **``fresnel_propagate_mft(E_in, z, wavelength, dx_in, dx_out, N_out, ...)``**
  -- new public propagator.  Single-FFT paraxial Fresnel on an
  arbitrary user-specified output grid via Bluestein chirp-Z transform.
  Same math as ``fresnel_propagate`` but the user picks ``dx_out`` and
  ``N_out`` independently of ``dx_in`` and ``z``.  Standard tool for
  focal-plane zoom (sample a tightly-focused output region at sub-pixel
  resolution without padding the input grid by the corresponding
  factor).  Also accepts ``centre_out=(x, y)`` to evaluate the output
  on an off-axis region of the focal plane.

* **``fraunhofer_propagate_mft(E_in, z, wavelength, dx_in, dx_out, N_out, ...)``**
  -- new public propagator.  Far-field counterpart to
  ``fresnel_propagate_mft``, drops the input-plane quadratic phase (the
  paraxial small-angle assumption).  Excellent for coronagraph and
  high-contrast imaging workflows -- sample the far-field at
  sub-lambda/D resolution around an off-axis stellar PSF without
  zero-padding the input pupil.  POPPy's ``apply_image_plane_fftmft``
  and prysm's ``focus_fixed_sampling`` are well-established equivalents
  in their respective ecosystems and the inspiration for this
  Lumenairy addition.

* **``angular_spectrum_propagate_mft(E_in, z, wavelength, dx_in, dx_out, N_out, ...)``**
  -- new public propagator.  Exact ASM (``exp(i*kz*z)`` with
  ``kz = sqrt(k^2 - kx^2 - ky^2)``) followed by a Bluestein chirp-Z
  inverse FT onto the user-specified output grid.  POPPy, prysm, and
  diffractio offer arbitrary-output-grid focal-zoom via their paraxial
  Fresnel propagators (which is the right choice for the imaging
  applications they're built for); this Lumenairy addition extends
  that capability to high-NA / strongly-diverging beams where the
  exact ASM kernel is preferred.  Same ``bandlimit=True`` default as
  ``angular_spectrum_propagate``.

* Internal helper module **``lumenairy/propagators/_bluestein.py``**
  with ``_bluestein_2d`` (the Bluestein chirp-Z primitive) and
  ``_bluestein_centred_2d`` (centred-index wrapper).  Used by all three
  MFT propagators; also reusable for future MFT-based work in
  ``analysis/`` (MTF, phase retrieval).

### Standardisation across all new propagators

Each new ``*_propagate_mft`` function follows the existing Lumenairy
propagator conventions:

* Backend dispatch identical to ``angular_spectrum_propagate``: NumPy
  CPU (with pyFFTW dispatch via ``_fft2`` / ``_ifft2``), CuPy GPU, JAX.
* Float32 / float64 controlled by ``DEFAULT_COMPLEX_DTYPE`` and the
  input dtype.
* JAX-traceable end-to-end (validated with ``jax.grad``).
* Carrier ``exp(i*k*z)`` preserved -- absolute-phase convention,
  consistent with the rest of Lumenairy and most of the Fresnel/ASM
  ecosystem.
* Live in ``lumenairy/propagators/propagation.py`` next to their
  same-grid siblings; exposed at top level (``la.fresnel_propagate_mft``,
  etc.) and added to ``__all__``.

### Performance

Bluestein chirp-Z transform achieves ``O((N+M) log (N+M))`` per axis
where ``N = N_in`` and ``M = N_out``.  Compared to the alternative
matrix-Fourier transform (``O(N^2 M^2)``):

| Setting | MDFT (CPU) | Bluestein/CZT (CPU) |
|---|---|---|
| 512^2 -> 128^2 | ~50-100 ms | **~10-30 ms** |
| 2048^2 -> 256^2 | ~2-5 s | **~80-200 ms** |

The Bluestein helper internally calls Lumenairy's ``_fft2`` / ``_ifft2``
dispatch, so pyFFTW is used automatically when available.  GPU and JAX
paths benefit from cuFFT / XLA acceleration of the same FFTs.

### Validation

| # | Test | Result |
|---|------|--------|
| 1 | ``apply_fresnel_curvature``: ``sign=+1`` then ``-1`` round-trips exactly | round-trip err 1.25e-16; mag unchanged; phase change non-trivial |
| 2 | ``apply_fresnel_curvature``: ``R=0`` and ``R=inf`` are identities | passes |
| 3 | ``fresnel_propagate_mft`` @ natural grid = ``fresnel_propagate`` | rel err **2.25e-14** |
| 4 | ``fraunhofer_propagate_mft`` @ natural grid = ``fraunhofer_propagate`` | rel err **2.09e-14** |
| 5 | ``angular_spectrum_propagate_mft`` @ same grid = ``angular_spectrum_propagate`` | rel err **2.35e-14** |
| 6 | Fresnel-MFT focal zoom matches Gaussian-beam ABCD formula (waist + amplitude) | 1.28% / 0.00% |
| 7 | Fraunhofer-MFT focal zoom matches Airy first-null ``1.22*lambda*z/D`` | exact (0.00%) |
| 8 | ASM-MFT central sub-window matches ASM reference | rel err **1.07e-14** |
| 9-11 | ``jax.grad`` of all three MFT propagators matches FD | rel err 2-5e-5 |

Total propagator-suite tests: **40 / 40 pass.**

### Backwards compatibility

``__all__`` grows from 391 to 395 entries (4 new public functions).
No existing API behaviour changes.

## [3.5.6] — 2026-05-07

CI hotfix + 14 audit-driven performance / accuracy improvements.

### CI hotfix

* `lenses.py`: missing ``from typing import ... Tuple`` import after
  the 3.5.5 split caused ``NameError: name 'Tuple' is not defined``
  at import on Linux/3.11-3.13.  Locally on Python 3.14 the lazy
  annotation evaluation hid the issue; fixed for all Python versions.
  This is the only reason 3.5.5 didn't reach PyPI cleanly.

### Performance & accuracy

| # | Item | Result |
|---|------|--------|
| 1 | **`trace_jax_with_params`** -- the audit's #1 lever | New JAX-array-aware trace that accepts ``radii`` / ``conics`` / ``aspheric_coeffs`` / ``thicknesses`` as differentiable JAX arrays.  Default kwargs reproduce ``trace_jax`` to roundoff; ``jax.grad`` through R1 matches finite differences to 1e-5 rel.  Unblocks design-parameter adjoint optimization that 3.5.5's ``make_lg_aberration_merit_jax`` only partially supported. |
| 2 | Newton iter cap reverted 8 -> 12 in ``apply_real_lens_traced`` | 3.5.5 dropped this based on an audit recommendation; benchmarks showed 0% speedup (active-mask early-exit dominates).  Restoring 12 protects outlier pixels that genuinely need 9-12 iters. |
| 3 | Adaptive Newton tolerance + warning | Surface non-convergence: when >1% of wave-grid pixels fail to converge to ``tol=0.01*dx`` within ``newton_max_iters`` iterations, emit ``RuntimeWarning`` (suppressed by ``on_undersample='silent'``).  Previously silently retained the last (possibly-wrong) Newton value. |
| 4 | Maslov upsample phase fix | ``apply_real_lens_maslov`` with ``output_subsample > 1`` previously line-by-line ``np.unwrap``'d the phase before zoom -- fragile near caustics.  3.5.6 interpolates the complex ``exp(i*phase)`` directly via cubic zoom of cos/sin, eliminating unwrap-induced seams. |
| 5 | JAX x64 auto-enable in ``fit_canonical_polynomials_jax`` | Was raising ``RuntimeError`` if x64 wasn't on; now emits a one-time warning and enables x64 itself.  Raises precision (single-precision lstsq gave 5% coefficient error and NaN gradients), never lowers it. |
| 6 | ``poly_order`` auto-bump in canonical fit | New ``auto_bump_threshold_waves`` kwarg.  When the fit residual exceeds the threshold, recursively retry at order+2 up to ``max_auto_poly_order``.  Helps cemented multi-element systems where the default order=6 under-fits. |
| 7 | ``set_pyfftw_planner('FFTW_MEASURE')`` | New API.  Takes ~1 s to plan but produces ~20% faster execution on the planned shape.  ``FFTW_PATIENT`` and ``FFTW_EXHAUSTIVE`` also supported. |
| 8 | Vectorised Vandermonde build in canonical fit | Replaces the per-basis-term Python loop with fancy-indexed elementwise product.  Same physics, cleaner code; ``lstsq`` still dominates runtime so wall-time is comparable. |
| 9 | Vectorised ``_evaluate_polynomial_4d`` (NumPy + xp) | Removes the per-basis-term Python loop in both the NumPy and xp variants.  Used in Maslov integration's hot path; ~3-5x faster on large grids. |
| 10 | ``design_optimize(precision='single')`` | New kwarg switches the default complex dtype to ``np.complex64`` for the duration of the optimization call (~2x FFT throughput, ~2x memory headroom).  Restored on return. |
| 11 | ``non_sequential_stray_light`` wrapper | Combines ``ghost_analysis`` + optional BSDF TIS into a single structured stray-light report.  Returns ghost paths, total ghost intensity, per-surface TIS, and a conservative stray-light fraction. |
| 12 | ``monte_carlo_tolerancing_linearized`` | Linearised tolerancing via per-knob FD sensitivity sweep + per-trial linear superposition.  ~3-6x faster than the full MC for typical specs (1 nominal + ~16-30 FD probes vs N_trials full propagations).  Accuracy degrades for large perturbations; not a final-sign-off tool. |

### Items deferred (not in 3.5.6)

* Pre-allocated buffers in merit eval -- modest impact (~5-15%), high refactor risk; skipped.
* Cached prescription parsing -- empirical 3.3 us per call, not a bottleneck; skipped.
* ``apply_real_lens_traced(use_gpu=True)`` benchmark -- already plumbed since 3.2.x; no CUDA available locally to benchmark.

### Validation

All 25 validation files pass.  +2 new tests in ``test_raytrace.py``
covering ``trace_jax_with_params`` (default-match-trace_jax + jax.grad
vs FD).

### Backwards compatibility

`__all__` grows from 384 to 391 (`set_pyfftw_planner`,
`non_sequential_stray_light`, `monte_carlo_tolerancing_linearized`,
`trace_jax_with_params`, `make_lg_aberration_merit_jax` -- the last
one was already in 3.5.5; counting kept honest now).

## [3.5.5] — 2026-05-06

Performance + structural cleanup release.

### Performance: 6 audit-driven optimisations

| # | Item | Result |
|---|------|--------|
| 1 | JAX adjoint gradient via `make_lg_aberration_merit_jax` | New convenience factory wraps `JaxMeritTerm` for the LG-aberration case.  Differentiable inputs (`wavelength`, `source_box_half`, `pupil_box_half`, `object_distance`, `w_s`, `w_p`) flow through `design_optimize`'s `jac='auto'` path automatically.  **Prescription-parameter differentiation (radii / conics / aspheric) remains FD-only**: gated on a `trace_jax`-with-JAX-array-prescription extension that's a separate roadmap item. |
| 2 | Cache canonical fit across multi-term merits | New `EvaluationContext._canonical_fit_cache`.  When a `CompositeMerit` contains several `LGAberrationMerit` terms with identical `fit_kwargs` (typical: one term per emitter class -- centre / edge / corner), the canonical fit is built once per merit eval and shared.  Validated **2.8x speedup** on a 3-term composite. |
| 3 | GPU FFT path for `apply_real_lens` | `apply_real_lens(use_gpu=True)` and CuPy auto-dispatch were already wired but the module docstring incorrectly said "CPU only".  Stale docstring fixed; lazy-load bug from 3.5.3 (`xp = cp` when `cp` was None) fixed. |
| 4 | Persistent worker pool for `apply_real_lens_traced` | New `lumenairy.close_worker_pool()` API.  The Newton-inversion `ProcessPoolExecutor` is now a module-level lazy singleton instead of being spawned-and-torn-down per call.  Validated **3x speedup** on subsequent calls (4.4 s -> 1.5 s after first call) -- amortises Windows-spawn cost across optimisation / tolerancing runs. |
| 5 | Newton iter count adaptation | Default `_NEWTON_MAX_ITERS` lowered from 12 to 8 in `apply_real_lens_traced`.  Empirically a no-op (the per-pixel `active`-mask early-exit already converges most pixels in <8 iters), but the lower default shaves the worst-case for outlier pixels.  Override via `apply_real_lens_traced(newton_max_iters=12)` if needed. |
| 6 | Pre-warm FFT plan cache | New `lumenairy.warmup_fft_plans(shapes, dtype=None, threads=None)` API.  Pre-builds pyFFTW plans for given shapes so the first ASM call at each shape doesn't pay the planning cost.  Validated **1.8x speedup** on first call after warmup at N=2048. |

### Structural: lenses.py split into 6 focused modules

`lenses.py` was 5,597 lines.  Split into:

| File | Lines | Contents |
|---|--:|---|
| `lenses.py` | **950** | Surface sag, grid-vs-aperture utilities, shared Chebyshev helpers, re-exports |
| `_lens_thin.py` | 576 | `apply_thin_lens`, `apply_spherical_lens`, `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply_axicon`, `apply_grin_lens` |
| `_lens_real.py` | 629 | `apply_real_lens` (analytic split-step ASM through glass) |
| `_lens_traced.py` | 2,098 | `apply_real_lens_traced` + `_Cheb2DEvaluator` + Newton inversion + worker pool helpers |
| `lenses_maslov.py` | 857 | `apply_real_lens_maslov` (phase-space Maslov propagator) |
| `_lens_jax.py` | 710 | `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax` + JAX helpers |
| **Total** | 5,820 | (small overhead from per-file headers + lazy-load shims) |

**`lenses.py` is now 83% smaller** (5,597 -> 950 lines).  All public names are re-exported so existing imports continue to work unchanged:

```python
from lumenairy.elements.lenses import apply_real_lens         # works
from lumenairy.elements.lenses import apply_real_lens_traced  # works
from lumenairy.elements.lenses import apply_real_lens_maslov  # works
```

The mid-file `from .. import raytrace as rt` workaround that the audit flagged is gone -- it now lives at the top of `lenses_maslov.py` where it belongs.

### Audit follow-ups not in 3.5.5

The "real" P1 (JAX adjoint through prescription parameters -- radii, conics, aspheric coefficients) requires a `trace_jax` extension that takes JAX-array surface params instead of static Python floats.  That's a multi-day refactor of every static `bool(np.isinf(R))` / static-conic branch in `_intersect_jax` / `_refract_jax`.  Tagged as a future item.

Stray-light pipeline integration, deeper pytest refactor, NA-aware vector source primitive, and detector ROIC modelling remain on the roadmap.

### Validation

All 25 validation files pass.  No new test files were added in this release; the existing physics tolerances are preserved bit-for-bit by the structural split (the function bodies were moved verbatim).

### Backwards compatibility

Every public API continues to work unchanged.  `__all__` grows from 380 to 384 (`make_lg_aberration_merit_jax`, `warmup_fft_plans`, `close_worker_pool`, plus the `tolerancing_report` from 3.5.4 that was already counted).

## [3.5.4] — 2026-05-06

Audit follow-up release.  Three audit-flagged gaps closed; one
quiet-failure mode in the propagator core is now visible.

### `caustic_diagnostic` -- "where do focuses live in this design?"

New analyzer that surfaces the Maslov machinery already present
in `apply_real_lens_maslov`.  Traces a small fan of rays through a
prescription, computes the Jacobian eigenvalues at sample planes
between every pair of refractive surfaces, and reports:

* `caustic_z` -- z coordinates of caustic / focal-point crossings.
* `maslov_index` -- total Jacobian-eigenvalue zero-crossings (the
  exact integer the Maslov-method propagator adds ``-pi/2``-per to
  the geometric phase).
* `det_J`, `chief_ray_height` -- per-z-sample arrays for plotting.

A companion `plot_caustic_diagnostic` helper produces a two-panel
figure showing det(J) and chief-ray height vs z, with caustic
crossings and surface positions annotated.

For an axisymmetric singlet at its expected BFL, the diagnostic
reports one caustic crossing at the focal point with Maslov index
2 (point caustic = both eigenvalues cross simultaneously); for a
tilted-input astigmatic system, it splits into two caustics
(sagittal + tangential) with Maslov index 2.

Closes the audit's "Maslov index / caustic crossing diagnostic
not surfaced to users" gap.

### `tolerancing_report` -- structured reporting for MC runs

New helper that turns the raw output of `monte_carlo_tolerancing`
or `monte_carlo_tolerancing_jax` into:

* Strehl summary (mean / std / p05 / p50 / p95 / min / max)
* Yield at standard thresholds (P(S > 0.5), 0.7, 0.8, 0.9, 0.95)
* Strehl-yield curve data (CDF arrays for plotting)
* *Optional* per-knob sensitivity ranking (correlation between
  perturbation magnitude and Strehl loss, ranked by importance) --
  pass `perturbation_spec=` and `trial_perturbations=` to enable

Output format selectable via `format='text'` (default, prints a
structured report) or `format='dict'` (returns a dict for further
processing).

Closes the audit's "tolerancing report generator" gap.

### Pytest entry point

Added `tests/test_validation_files.py` plus a `[tool.pytest.ini_options]`
block in `pyproject.toml`.  Pytest now parametrizes over each of
the 25 validation files and runs them as subprocesses, giving:

* per-test isolation (`pytest -x` for fail-fast)
* filtering (`pytest -k 'asymptotic'` to run a subset)
* parallel execution via pytest-xdist (`pytest -n auto`)
* JUnit XML for CI (`pytest --junitxml=results.xml`)
* coverage hooks (`pytest --cov=lumenairy`)

Without refactoring the existing `validation/_harness.py`
infrastructure -- `validation/run_all.py` continues to work
unchanged.  This is a thin pytest layer; a deeper refactor (each
``t_*`` function as a parametrized pytest test) is on a future
roadmap.

### Core-path silent-failure fix in `propagators/propagation.py`

The audit flagged three `except Exception: pass` blocks in
`propagation.py`'s core path (lines 250, 599, 1626 in the 3.5.2
codebase).  All three are now visible:

* pyfftw cache reset failures in `reset_fft_backend` and after a
  per-shape FFT failure now emit `RuntimeWarning` instead of
  swallowing silently.  Users no longer hit a "pyFFTW used the
  wrong cache" Heisenbug without diagnostic.
* The verbose-mode kernel-max printout in
  `propagate_rayleigh_sommerfeld` (which can fail under JAX
  tracing because `xp.max + float()` doesn't work on abstract
  arrays) now prints `<unavailable: TraceError>` with the
  exception class name instead of silently producing no output.

The remaining 169 `except Exception: pass` blocks across the
package are mostly Qt/PySide UI defensive disconnects (125 of
them) where silent fall-through is correct; the non-UI blocks
have been triaged and confirmed defensive.

### Validation

All 25 validation files pass.  +4 new tests in `test_analysis.py`:
caustic_diagnostic singlet-focus check, caustic_diagnostic
flat-plate-no-caustic check, tolerancing_report text-format
non-empty check, tolerancing_report dict-yields check.

### Backwards compatibility

`__all__` grows from 377 to 380 (`caustic_diagnostic`,
`plot_caustic_diagnostic`, `CausticDiagnostic`, `tolerancing_report`).
No existing API changes.

## [3.5.3] — 2026-05-06

Audit-driven cleanup release.  No behaviour changes for any
existing supported call; new exported names and faster startup.

### Lazy-loading of heavy optional dependencies

`import lumenairy` previously eagerly loaded JAX, CuPy, h5py,
pyfftw, numexpr, matplotlib, astropy, refractiveindex, and PIL --
roughly 2.3 s on a development box even when the user only
wanted the NumPy ASM path.  All ten dependencies are now lazy:

* `find_spec`-based availability checks at import time replace
  unguarded `try: import X` blocks in `backend/array.py`,
  `backend/scipy.py`, `backend/fft.py`, `propagators/propagation.py`,
  `elements/lenses.py`, `elements/doe.py`, `glass.py`,
  `analysis/plotting.py`, `io/storage.py`, `sources/core.py`.
* `_ensure_*_loaded()` helpers (or per-module `_get_*()`
  accessors) import the actual package on first use.
* The `JAX_AVAILABLE` / `CUPY_AVAILABLE` / `_H5PY_AVAILABLE` /
  etc. module-level constants are preserved -- the ~60 callers
  across the package that branch on them require no change.

Result: `import lumenairy` measured at **0.60 s** end-to-end,
**0 heavy deps** loaded eagerly (down from 10).  Roughly **3.85x
faster startup**.  Net: every CLI script and notebook that uses
the library starts up faster, and JAX is no longer pulled in
for purely-NumPy workflows.

### Canonical propagator argument order

Three of the four end-to-end propagators historically used
`(E_in, dx, *, z, wavelength, ...)`; only `angular_spectrum_propagate`
and `propagate_huygens_fresnel_freespace` used the canonical
`(E_in, z, wavelength, dx, ...)`.  3.5.3 adds the canonical-order
forms as separate names, leaving the legacy signatures untouched
for backwards compatibility:

* `lumenairy.propagate_gbd(E_in, z, wavelength, dx, **kwargs)` --
  delegates to `propagate_gbd_freespace`.
* `lumenairy.propagate_hfpi(E_in, z, wavelength, dx, *,
  aperture_radius, z_aperture_to_output, n_paths, **kwargs)` --
  delegates to `propagate_hfpi_freespace_aperture`.

The legacy `propagate_gbd_freespace` and
`propagate_hfpi_freespace_aperture` continue to work; their
docstrings now point users at the canonical-order versions.

### New analysis utilities

* `lumenairy.coupling_efficiency(E, mode, dx, dy=None)` --
  classical mode-overlap coupling efficiency for fiber-coupling /
  receiver-mode-matching applications.  Returns
  `|<E|mode>|^2 / (<E|E>*<mode|mode>)` in [0, 1].  Validated
  against the analytic 2-D Gaussian-to-Gaussian overlap
  ``(2 w1 w2 / (w1^2 + w2^2))^2``.
* `lumenairy.M2(E, dx, wavelength, dy=None)` -- ISO 11146
  beam-quality factor at a single plane, with phase-curvature
  correction via the Wigner cross-term so the result is
  invariant under propagation.  Returns `(M2_x, M2_y)`.  A
  fundamental Gaussian gives 1.0 to grid-sampling precision; a
  curved (non-waist) Gaussian still gives 1.0; a multi-mode beam
  gives M2 > 1 along the spread axis.

Both have validation tests in `test_analysis.py`.

### Library export bug fixes (carried over from 3.5.2 work)

* `export_zemax_zmx` and `export_zemax_lens_data` now honour
  per-surface `semi_diameter` from the prescription dict
  (previously emitted only the global `aperture_diameter / 2`
  for every row, dropping aperture-override information).
* `export_zemax_zmx` now emits aspheric coefficients via
  `TYPE EVENASPH` + `PARM` rows instead of silently dropping
  them.  Coefficient unit conversion is m^(1-power) -> mm^(1-power).
* `export_zemax_lens_data` `extra_notes=` now correctly handles
  a list-of-strings (a string with newlines was previously
  iterated character-by-character into one-character comment
  lines).

### Wiki

Four new pages (filling the audit's documentation gaps):

* [Huygens-Fresnel Path Integration](https://github.com/travaj24/LumenAiry/wiki/Huygens-Fresnel-Path-Integration)
* [Gaussian Beamlet Decomposition](https://github.com/travaj24/LumenAiry/wiki/Gaussian-Beamlet-Decomposition)
* [Multi-Huygens-Surface and Patches](https://github.com/travaj24/LumenAiry/wiki/Multi-Huygens-Surface-and-Patches)
* [Validation and Accuracy](https://github.com/travaj24/LumenAiry/wiki/Validation-and-Accuracy)
  (single-page index of every quantitative tolerance in the suite)

### Validation

All 25 validation files pass; +6 new tests in `test_analysis.py`
covering `coupling_efficiency` (self-overlap, 2x-wider analytic
match, orthogonal-tilt zero) and `M2` (fundamental-Gaussian
unity, curvature invariance, two-spot-beam > 1).

### Known follow-ups (not in 3.5.3)

* Migration of the validation harness to pytest with proper
  fixtures.
* Triage of the ~178 `except Exception: pass` blocks across the
  package -- audit flagged these as future Heisenbugs.
* Surfacing the Maslov index / caustic-crossing diagnostic to
  end users.
* Stray-light pipeline integration (`BSDFModel` +
  `enumerate_ghost_paths` + `coatings.py`).

## [3.5.2] — 2026-05-06

### All four 3.5.1 reserved JAX stubs are now real implementations

3.5.1 added seven JAX-companion functions: three real, four reserved
stubs that raised `NotImplementedError`.  3.5.2 lands the four
remaining functions as real, validated, ``jax.grad``-compatible
implementations.

**`lumenairy.apply_real_lens_traced_jax`**

Per-pixel ray-traced phase screen built from `trace_jax` + Newton
inversion of the entrance->exit map (Chebyshev tensor-product fit
for the inverse interpolant).  Default cheb_order=10 (66 basis
terms) reaches sub-nm OPD residual on typical refractive lenses.
Output is `E_in * mask * exp(i k0 OPD)` (thin-OPD-screen
treatment); pass `amplitude='analytic'` for the NumPy-version's
diffractive-amplitude leg via callback.  ``jax.grad`` flows from
the output field back through ``E_in``; the prescription dict is
treated as static (same constraint as `trace_jax`).  Validated
against `apply_real_lens_traced` to ~0.7 nm RMS / 1 nm peak OPD on
a moderate singlet, and ``jax.grad`` matches finite differences to
5e-7 relative.

**`lumenairy.apply_real_lens_maslov_jax`**

Same structure as `apply_real_lens_traced_jax` plus a Maslov-index
correction: counts Jacobian-determinant sign flips along the
radial path from origin to each entrance pixel and adds
``-pi/2 * count`` to the geometric phase.  Extends valid OPD
modelling into caustic / focal-region neighbourhoods.  For
non-caustic geometries the Maslov index is zero everywhere and
the output matches `apply_real_lens_traced_jax` exactly.

**`lumenairy.fit_canonical_polynomials_jax`**

JAX-traceable canonical polynomial fit.  Uses `trace_jax` for the
sample-collection ray bundle and `jnp.linalg.lstsq` for the 4-D
Chebyshev tensor-product solve.  Returns a `CanonicalPolyFit`
populated with JAX arrays (the dataclass's `eval_phi_xp` /
`eval_s1_xp` methods preserve the gradient graph end-to-end into
`aberration_tensor_lg00_jax` and `solve_envelope_stationary_jax_ift`).
Coefficients agree with the NumPy fit to 1e-7 relative on a
moderate singlet; ``jax.grad`` w.r.t. ``source_box_half`` matches
finite differences to ~5e-3 relative (lstsq backward through the
fit normaliser is the limiting factor).  Requires JAX x64 mode --
single-precision lstsq + Chebyshev tensor product gives ~5%
coefficient error and NaN gradients, so the function raises if
called without `jax.config.update('jax_enable_x64', True)`.

**`lumenairy.monte_carlo_tolerancing_jax`**

JAX-accelerated trial sweep.  Per-trial perturbation generation
stays in NumPy (``apply_perturbations`` mutates a Python
prescription dict); the wave-leg propagation routes through either
`apply_real_lens_traced_jax` (default) or NumPy `apply_real_lens`
(``wave_propagator='real_lens'``), and the through-focus scan uses
`through_focus_scan_jax`.  Identical Strehl distribution as the
NumPy version on the same RNG seed; ~25% faster on the validation
case (5 trials, 11-point z-scan, N=256) thanks to the fused
per-z propagation.

### Implementation note

* `apply_real_lens_traced_jax` clamps the entrance ray-launch radius
  to 1.02x the aperture (the NumPy reference uses 1.5x).  The
  larger over-margin is unsafe in the JAX path because `trace_jax`
  returns finite OPL values for rays whose geometric trajectory
  would have negative edge-thickness through the glass (their
  intersection points sit beyond where the next surface vertex
  lies), where the NumPy `trace` correctly stops them.  Marginal
  wave-grid pixels at the very edge that map back to entrance
  positions slightly beyond aperture/2 are now zeroed by the
  aperture mask -- the same final-zeroing the NumPy version
  applies anyway.

### `__all__`

Size unchanged at 373.  Behaviour change: four entries that
previously raised `NotImplementedError` now work.

### Validation

All 25 validation files pass.  Two new tests in `test_lenses.py`
(`apply_real_lens_traced_jax` matches NumPy OPD; ``jax.grad`` vs
FD), one in `test_lenses.py` (Maslov vs traced for non-caustic),
two in `test_asymptotic.py` (`fit_canonical_polynomials_jax`
matches NumPy; ``jax.grad`` finite), one in `test_analysis.py`
(`monte_carlo_tolerancing_jax` matches NumPy on shared RNG).

## [3.5.1] — 2026-05-05

### Additive JAX paths across analysis + system + propagators

Seven JAX-companion functions added alongside their NumPy originals
(none replace existing functionality; all are opt-in via the `_jax`
suffix).  Three are real implementations; four are reserved stubs
documenting the planned interface and raising
`NotImplementedError` if called.

**Real implementations:**

- `lumenairy.through_focus_scan_jax` -- JAX-batched per-z propagation
  via JAX-traceable `angular_spectrum_propagate`.  Same return
  contract as `through_focus_scan`.  CPU runtime is comparable to the
  NumPy version (pyFFTW-based ASM is fast on CPU); GPU runtime can
  be 5-15x lower because the FFT batch fuses into one cuFFT call.
- `lumenairy.gerchberg_saxton_jax`,
  `lumenairy.error_reduction_jax`,
  `lumenairy.hybrid_input_output_jax` -- JAX-jit'd phase-retrieval
  iterations using `jax.lax.fori_loop`.  Whole iteration loop runs
  in one fused JIT kernel.  Return contracts match NumPy versions.
- `lumenairy.propagate_through_system_jax` -- element-by-element walk
  with per-element JAX dispatch for `propagate`, `lens`, `aperture`,
  `mask`.  Element types without a JAX path
  (`spherical_lens` / `aspheric_lens` / `real_lens` / `mirror`)
  fall back to NumPy at the element boundary -- the field is
  converted to NumPy for that element only, then back to JAX.

**Reserved stubs (raise `NotImplementedError`):**

- `lumenairy.fit_canonical_polynomials_jax` -- needs careful rewrite
  of the sample-collection ray trace + Chebyshev least-squares solve
  in JAX.  Existing fits remain consumable by JAX downstream paths
  (`aberration_tensor_lg00_jax`, `solve_envelope_stationary_jax_ift`)
  via their `eval_phi_xp` / `eval_s1_xp` methods.
- `lumenairy.apply_real_lens_traced_jax`,
  `lumenairy.apply_real_lens_maslov_jax` -- planned vmap over per-pixel
  ray launches via `trace_jax`.  GPU users get 20-50x speedup from
  the vmap; multi-core CPU users already get most of that from the
  existing `n_workers` parallelism in the NumPy versions, so the JAX
  rewrite is medium-priority.
- `lumenairy.monte_carlo_tolerancing_jax` -- vmap over trial seeds.
  Blocked on `apply_real_lens_traced_jax` for the per-trial wave
  leg.

All seven are now exposed in `__all__` (size 373 entries).  Calling a
stub raises a clear `NotImplementedError` with a pointer to the NumPy
version.

### `solve_envelope_stationary_jax_ift` — JAX-grad-friendly Newton solver

New library function `lumenairy.solve_envelope_stationary_jax_ift`
(also exposed at `lumenairy.propagators.asymptotic`).  Wraps the
existing NumPy `solve_envelope_stationary` in a `jax.custom_vjp`
that uses the **implicit function theorem** for the backward pass:

- Forward: 15-iteration Gauss-Newton in `jax.lax.fori_loop` (fixed
  iter count, JIT/vmap-friendly).
- Backward: a single 2x2 linear solve `[∂F/∂v]ᵀ λ = grad_v` followed
  by `−λᵀ ∂F/∂θ` for each differentiable input.  No autograd
  unrolled through the iteration.

The IFT gradient is exact at the converged fixed point regardless
of `n_iter`; the computational graph stays small.  Differentiable
w.r.t. `s2`, `source_point`, `w_s`, `w_p`, and `v2_centre`.  The
`fit` is treated as a non-differentiable closure (its coefficients
come from a NumPy least-squares step that isn't part of the JAX
graph).

**Lazy JAX:** module imports cleanly without JAX installed; the
`jax.custom_vjp` decoration runs (and is cached) on the first call.
Identical pattern to `aberration_tensor_lg00_jax`,
`propagate_modal_asymptotic_lg00_jax`, and `trace_jax`.

### Validation

Two new tests in `test_asymptotic.py`:
- Forward matches the NumPy solver on an off-axis source point to
  4.6e-12 max-error.
- IFT backward matches finite-difference gradient of `dv*₀+v*₁/d(source_x)`
  to rel < 1e-3 (single-precision JAX floor; would be ~1e-10 in
  float64).

All 25 validation files green.

## [3.5.0] — 2026-05-05

Three sessions of feature work landed under 3.5.0: completing the
deferred JAX / asymptotic / trace items, then a sweep of "synergy"
upgrades that wire the new propagators into the optimize / dispatch /
storage layers, then a final pass of API harmonization.  All 25
validation files green throughout.

### JAX-traceable ray trace gains aspheric / aperture / DOE support

`lumenairy/raytrace/jax_trace.py`:

- **Aspheric Newton intersect** via `jax.lax.fori_loop` (8 fixed
  iterations).  Surface conic + even-aspheric coefficients are
  honoured; the closed-form spherical/flat path remains for surfaces
  without aspherics.  Intersections fully JIT-able and grad-able.
- **Aperture clipping** via the `alive` mask -- rays whose
  intersection point falls outside `semi_diameter` are vignetted
  forward.
- **DOE order kicks**: `trace_jax(..., surface_diffraction={i: (m_x,
  m_y, period_x, period_y)})` shifts direction cosines by
  `m * lambda / period` and adds the linear OPL term at the
  intersection point.  Evanescent orders kill the ray.
- **Bidirectional RayBundle <-> JaxRayState conversions**:
  `RayBundle.to_jax_state()`, `raybundle_to_jax_state(rb)` (mirror
  of existing `jax_state_to_raybundle(state, wavelength=...)`).

### LG aberration tensor + modal asymptotic on JAX

`lumenairy/propagators/asymptotic.py`:

- `aberration_tensor_lg00_jax(fit, s2_image, v_star, ...,
  return_result=False)` -- closed-form L_{(0,0),(0,0)} coefficient.
  Newton solve for `v_star` is performed externally; the JAX graph
  itself is closed-form and differentiable wrt fit coefficients,
  s2_image, v_star, source_point, and waist parameters.
  `return_result=True` returns a `JaxAberrationTensorResult`
  NamedTuple mirroring the NumPy result's `.L` / `.output_modes` /
  `.w_o` shape (with `.L` a (1, 1) JAX array).
- `propagate_modal_asymptotic_lg00_jax(fit, s2_grid_x, s2_grid_y,
  v_star_grid, ...)` -- vmap'd per-pixel evaluator.  Takes a
  pre-solved `v_star` grid (typically from the NumPy
  `solve_envelope_stationary` warm-start chain); skipping the
  per-pixel Newton keeps the entire evaluator JIT/grad-friendly.

### Multiple Huygens Surface (MHS) framework

New module `lumenairy/propagators/mhs.py`:

- `HuygensSurface(z, Ny, Nx, dx, centre, label)` dataclass for a
  sample plane.
- `MhsSubdomain(propagator, in_surface, out_surface, kwargs, label)`
  -- a single plane-to-plane operator.
- `MhsPipeline([sub_a, sub_b, ...])` validates the surface-chain
  consistency, then `pipeline.run(E_in, ...)` walks the chain.
- Convenience builders: `asm_subdomain`, `aperture_subdomain`,
  `gbd_freespace_subdomain`, `prescription_subdomain` (the last
  routes through `la.propagate(method=...)`).
- **`MhsPipeline.from_prescription(prescription, wavelength=, dx=,
  Ny=, Nx=, pre_distance=, post_distance=, method='gbd')`**
  one-call ASM -> prescription -> ASM chain builder.
- **Storage hooks**: `pipeline.run(checkpoint=cb, store=path,
  label_prefix='mhs')` fires per-subdomain callback and optionally
  streams every plane through `la.append_plane` for replay.

### Top-level dispatcher: smarter auto + MHS routing + result wrapper

`lumenairy/propagators/dispatch.py`:

- **`'mhs'` added to `VALID_METHODS`**: `la.propagate(method='mhs',
  subdomains=[...])` or `pipeline=...` runs the MHS chain inline.
- **Smarter `_auto_select_method`**: when given a prescription the
  dispatcher now inspects surfaces for aspheric coefficients (->
  `'gbd'`), DOE / grating events (-> `'hfpi'`), and finite hard
  apertures (when `accuracy='accurate'` -> `'hf'`).  New `accuracy`
  hint (`'fast'` | `'balanced'` | `'accurate'`).
- **Through-prescription routing for asymptotic** -- the dispatcher
  now builds a `CanonicalPolyFit` from the prescription on the fly
  if the user didn't supply one.  Pre-existing signature mismatch
  fixed.
- **`return_result=False` opt-in flag**: when True, wraps the
  output in a `PropagationResult` carrying `.field`, `.dx`,
  `.wavelength`, `.method`, `.metadata`.  Default behaviour is
  unchanged (bare ndarray) -- zero-overhead fast loops still work
  exactly as before.

### Unified `PropagationResult`

New module `lumenairy/propagators/result.py`:

`PropagationResult(field, dx, wavelength, z, method, history,
metadata, intermediates)` -- one container shared by `propagate()`,
`MhsPipeline.run(return_result=True)`, and
`propagate_through_system(return_result=True)`.

- `.field` is the exit-plane field; `.history` is a list of
  `(label, field, dx)` tuples for plane-walking propagators.
- Tuple-unpacks as `(field, intermediates)` -- a drop-in for
  callers that did `E, intermediates = propagate_through_system(...)`.
- `np.asarray(result)` returns the field, so it slots in wherever a
  bare ndarray is accepted.
- `.to_source()` wraps the result back into a `Source`.

Every wrapping is **opt-in via `return_result=True`** -- existing
return shapes are preserved everywhere.

### `Source` class

New `lumenairy.sources.Source` dataclass bundles
`(E, dx, wavelength, source_point, name)` with chainable
`.propagate(method='auto', z=..., prescription=..., **kwargs)`:

- Returns another `Source` so chains read like English:
  `Source.gaussian(...).propagate(method='asm', z=10e-3).
  propagate(method='gbd', prescription=p)`.
- Class-method factories: `Source.gaussian`, `Source.plane_wave`,
  `Source.point_source`, `Source.top_hat`, `Source.fiber_mode`.
- Wavelength and source_point are inherited automatically; the name
  is extended with `->{method}` for trace-ability.

### Optimize layer extensions

`lumenairy/optimize/core.py`:

- **`wave_propagator` parameter** on `design_optimize`: selects the
  wave-leg propagator from
  `'real_lens'`/`'gbd'`/`'hf'`/`'hfpi'`/`'asymptotic'`.  Keeps every
  existing merit term as-is (they read `ctx.E_exit`); the optimizer
  can now drive any of the modern propagators as the wave model.
- **`WAVE_PROPAGATOR_REGISTRY` + `register_wave_propagator(name,
  fn)`**: the if/elif dispatch is now a registry.  Users can plug in
  custom propagators (e.g. an external simulator) and use them via
  `wave_propagator=name`.
- **`JaxMeritTerm`** -- differentiable merit term wrapping a
  JAX-traceable callable.  Two modes:
  - `JaxMeritTerm(fn, ...)` -- forward-only; SciPy's FD gradient
    handles the rest.
  - `JaxMeritTerm(fn, build_args=lambda x: (...))` --
    differentiable in x.  `gradient_at_x(x)` returns a NumPy array
    via `jax.grad`.
- **`jac='auto' | 'fd' | callable`** parameter on `design_optimize`:
  when at least one `JaxMeritTerm` has `build_args`, an analytic
  Jacobian is assembled (analytic for JAX merits + FD for the
  rest) and passed to SciPy as `jac=`.  Test confirms 1-parameter
  problem converges to err=6.87e-9 in 30 iterations.
- **`plane_logger=fn(iter, ctx)`** parameter on `design_optimize`:
  callback fires after every merit evaluation -- useful for
  streaming intermediate `ctx.E_exit` / OPD / prescription state to
  a unified store.
- `EvaluationContext.x` -- new field carrying the current parameter
  vector so JaxMeritTerm can route through its `build_args(x)`
  consistently.

### Unified aberration analysis

New module `lumenairy/analysis/aberration.py`:

- `aberration_summary(prescription, wavelength, ...)` -- one call
  returns a dataclass carrying summed Seidel coefficients +
  per-surface breakdown + EFL/BFL + LG aberration tensor.
- **`differentiable=True`** flag routes the LG-tensor branch through
  `aberration_tensor_lg00_jax`; the result's `.lg_tensor.L` is a
  (1, 1) JAX array (differentiable via `jax.grad`).
- `format_aberration_summary(summary, units='mm')` -- pretty-printer.

### Storage and replay

`lumenairy/io/storage.py` and `lumenairy/system.py`:

- **`replay_run(filepath, *, label_prefix=None, wavelength=None,
  method=None)`**: reads every plane from a Zarr / HDF5 store
  written by an MHS / system / design_optimize run and returns a
  `PropagationResult` with `.history`.  Closes the loop on stored
  runs -- a single call can replay, plot, or diff without re-running
  the simulation.
- **`propagate_through_system(checkpoint=fn, store=path,
  label_prefix='system', return_result=False)`**: per-element
  callback + optional Zarr / HDF5 streaming.  Backward-compatible
  `(E_out, intermediates)` tuple is the default return.

### `__all__` reorganized into 10 user-journey tiers

`lumenairy/__init__.py` -- 358-entry `__all__` is now grouped:

1. Build a system (sources, prescriptions, lenses, elements)
2. Propagate (dispatcher + propagator families)
3. Trace (geometric + JAX-traceable)
4. Analyze (aberration_summary, through-focus, tolerancing,
   coherence, detector, interferometry, phase retrieval, ghost)
5. Optimize (parameterizations, merit terms, design_optimize,
   wave-propagator registry)
6. Asymptotic / LG aberration tensor
7. Specialized physics (RCWA, BSDF, vector diffraction, coatings)
8. I/O (HDF5/Zarr storage, prescription formats, code-gen)
9. Plotting
10. Infrastructure (backend, memory, progress, JAX flag, precision)

No entries removed; reorder + clearer grouping.

### Examples directory

New `examples/` directory with five end-to-end runnable scripts:

| File | What it shows |
|------|---------------|
| `01_basic_propagation.py` | Source + free-space + thin-lens + propagate-to-focus |
| `02_design_optimization.py` | design_optimize with FocalLength + Strehl merits |
| `03_high_fidelity_wave.py` | wave_propagator: real_lens vs GBD vs HF on the same prescription |
| `04_jax_differentiable.py` | JaxMeritTerm + jac='auto' (analytic JAX gradients into SciPy) |
| `05_mhs_pipeline_with_replay.py` | MhsPipeline.from_prescription + checkpoint logging + replay_run |

### Pre-existing bug fixes

- `propagate_huygens_fresnel_through_prescription` was passing
  `source_lg_amps`/`pupil_lg_amps`/`output_grid` to
  `propagate_modal_asymptotic` (legacy kwarg names that no longer
  exist).  Fixed to use the current
  `source_amplitudes`/`pupil_amplitudes`/`s2_grid_x`/`s2_grid_y`.
- The dispatcher's `'asymptotic'` branch was calling
  `propagate_modal_asymptotic(E_in, prescription, dx, ...)` --
  wrong signature.  Fixed to build a fit from the prescription (or
  accept a pre-built one via `wave_propagator_kwargs={'fit': ...}`).

### Validation suite expansion

A targeted survey of the test suite identified gaps in physics-
correctness coverage, cross-functional integration, and quantitative
JAX/NumPy comparison.  ~32 new tests were added across the suite to
close those gaps; all 25 test files remain green.  Highlights:

- **Cross-method physics agreement.** GBD free-space matches ASM on
  collimated Gaussians (rel < 1%); MhsPipeline single-ASM matches
  direct ASM call to numerical zero; ASM NumPy vs JAX value
  equivalence (rel < 5e-3, JAX float32).
- **Power conservation.** ASM, Fresnel (with grid-rescaling fix),
  GBD free-space, MHS chains, and `apply_real_lens` on a clear
  aperture all verified.
- **JAX gradients vs finite differences.** ASM, `trace_jax`, and
  `aberration_tensor_lg00_jax` now have quantitative grad-vs-FD
  checks (instead of just `isfinite`).  ASM grad rel < 4e-5;
  aberration_tensor rel < 2e-3 (off-axis).
- **Through-prescription propagation.** GBD-through-prescription
  peak intensity verified to land at the system BFL.
- **Smarter dispatcher auto-mode.** Aspheric-routing, hard-aperture
  + accuracy='accurate' routing, and empty-surfaces fallback all
  tested.
- **PropagationResult interop.** Tuple-unpacking, `np.asarray()`
  array-protocol, `.to_source()` round-trip all covered.
- **Replay round-trip.** HDF5 + Zarr (when installed) both verified
  to reconstruct the stored field history.
- **Wave-propagator registry.** End-to-end `design_optimize` with
  `wave_propagator='gbd'` and `'hf'` confirmed to converge to
  target EFL.
- **Physics-law correctness.**  Lambertian BRDF = rho/pi for all
  upper-hemisphere directions (closed-form check).  Lambertian
  hemispheric sampling: <cos(theta)> = 2/3 (50k samples within
  3e-4 of analytic mean).  Ghost-path focus_z positions verified
  finite + distinct on a Thorlabs achromat (3 ghosts at 26.7,
  37.1, 46.9 mm).  Stokes S1/S0 preserved through ASM
  propagation.  Through-focus Strehl symmetric about best focus
  for an aberration-free thin-lens.  Richards-Wolf reduces to
  scalar at low NA (`|Ez|^2 / |Ex|^2 << 1`).
- **Tolerancing statistics.**  Monte-Carlo seed reproducibility
  verified (max diff = 0).  Strehl-peak std grows with
  perturbation magnitude (5e-5 tilt -> sigma=0.017; 5e-3 tilt
  -> sigma=0.193).

The expansion uncovered and fixed the two pre-existing bugs noted
above (`propagate_huygens_fresnel_through_prescription` and the
dispatcher's `'asymptotic'` branch).

## [3.4.0] — 2026-05-05

### Stage 2-4 features added (prescription paths, variance reduction, vectorial)

The new propagators introduced earlier in 3.4.0 now operate on full
optical prescriptions, not just free-space scenarios:

- **`propagate_hfpi_through_prescription`** -- HFPI walks a sequential
  prescription via :func:`lumenairy.raytrace.trace`, accumulates OPL
  into the path complex weights, and re-emits secondary HF sources
  at every diffracting surface (auto-detected from finite
  ``semi_diameter``).
- **`propagate_gbd_through_prescription`** -- paraxial GBD via
  system ABCD evolution.  Each beamlet's complex Q-parameter and
  base ray transform via the prescription's
  ``system_abcd_prescription`` matrix.  Adds the
  ``apply_abcd_to_beamlets`` primitive.
- **`propagate_huygens_fresnel_through_prescription`** -- Van-Vleck
  HF for prescriptions via the existing asymptotic saddle-point
  machinery (``method='asymptotic'``).  Direct 2-D quadrature
  variant (``method='direct'``) currently raises ``NotImplementedError``;
  it requires a new HF-form polynomial fit ``Phi(s1, s2)`` and is
  tracked as future work.

The top-level dispatcher (``propagate(method='auto'|'gbd'|'hfpi'|'hf')``)
now routes prescription-bearing calls to all of these.

**Variance reduction:** ``init_paths_stratified`` partitions the
source-pixel index x forward-cone direction sphere into
equal-solid-angle strata and samples one path per stratum, reducing
the variance of HFPI Monte-Carlo estimates by 2-10x for
smooth-amplitude / smooth-OPL systems.

**Subaperture asymptotic:** ``propagate_subaperture_asymptotic``
decomposes the source plane into overlapping patches, fits a local
polynomial per patch, propagates each, and recombines the per-patch
output fields with a partition-of-unity weighting.  Restores
deterministic-asymptotic accuracy for wide-field / high-NA systems
where a single global polynomial fit underflows at the box edges.

**Vectorial HFPI:** new module ``lumenairy.propagators.vectorial_hfpi``
extends scalar HFPI with Jones polarization vectors (Ex, Ey) per
path.  ``VectorPathBundle`` carries the polarization state;
``init_vector_paths_from_field``, ``propagate_vector_to_plane``,
``apply_vector_aperture_diffraction``, ``accumulate_vector_to_grid``,
and end-to-end ``propagate_vector_hfpi_freespace_aperture`` are
the public API.  Required for high-NA imaging (NA > 0.3) where
polarization rotates strongly across the focal plane.

### Final-batch deferred items (5 items)

The 5-item deferred list from earlier in the 3.4.0 development
cycle is now implemented:

**Item 1 -- Tensor-product Chebyshev HF**: New ``HFPolyFit`` (4-D
Chebyshev tensor product fit of ``Phi(s1, s2)``) and
``fit_hf_polynomials`` / ``propagate_hf_chebyshev_quadrature``.
The deferred ``method='direct'`` path of
``propagate_huygens_fresnel_through_prescription`` is now wired
to the direct 2-D quadrature with the analytical Van Vleck
density factor.

**Item 2 -- JAX backend in propagation**: ``angular_spectrum_propagate``,
``fresnel_propagate``, ``fraunhofer_propagate``, and
``rayleigh_sommerfeld_propagate`` now accept JAX arrays and stay
in the JAX backend.  All four are differentiable via ``jax.grad``
and JIT-able via ``jax.jit``.  The pyFFTW > scipy.fft > numpy.fft
priority chain is preserved for NumPy inputs; JAX uses ``jnp.fft``
(XLA).

**Item 3 -- JAX backend in asymptotic** (polynomial eval):
``CanonicalPolyFit.eval_phi_xp`` / ``eval_s1_xp`` and
``HFPolyFit.eval_phi_xp`` use backend-aware Chebyshev evaluation
so the OPL polynomial surfaces are differentiable on JAX inputs.
The full ``propagate_modal_asymptotic`` per-pixel Newton solver
remains NumPy-only (genuine multi-week refactor).

**Item 4 -- JAX-traceable trace**: New module
``lumenairy.raytrace.jax_trace`` with ``trace_jax``, a functional
ray-trace via ``jax.lax``-friendly surface walk.  Supports
spherical and flat refractive surfaces; aspherics, apertures
killing rays, and DOE order kicks are intentionally not in scope
(use the NumPy ``trace`` for those).  ``jax.grad(trace_jax_loss)``
returns finite gradients of OPD w.r.t. ray launch coordinates.

**Item 5 -- MHS framework**: New module
``lumenairy.propagators.mhs`` with ``HuygensSurface``,
``MhsSubdomain``, and ``MhsPipeline`` classes.  Compose multiple
subdomain propagators (ASM, GBD, HFPI, prescription-based, hard
aperture) into a single chain via Huygens-surface field
reconstruction.  Convenience builders: ``asm_subdomain``,
``aperture_subdomain``, ``gbd_freespace_subdomain``,
``prescription_subdomain``.

### Original 3.4.0 release notes (continued)

### Major — Multi-backend foundation, new propagators, full reorg

This release establishes first-class **NumPy / CuPy / JAX** backend
support, adds four new propagator modules, introduces a top-level
smart-method dispatcher, and begins reorganising the library into
subpackages.  Existing user code is unchanged: every previously
top-level import (e.g. ``from lumenairy import angular_spectrum_propagate``,
``from lumenairy.propagation import X``, ``from lumenairy.asymptotic
import Y``) continues to work via thin shim re-exports.

### Subpackage reorganisation (complete)

The full library has been reorganised into eight thematic
subpackages.  Every previous top-level import path is preserved
through thin re-export shims, so user code continues to work
unchanged.

- **`lumenairy.backend`** -- numerical-backend dispatch
  (``array_namespace``, FFT / RNG / scipy compat, CPU helpers).

- **`lumenairy.propagators`** -- diffraction-propagator family
  (``propagation``, ``asymptotic``, ``gbd``, ``hfpi``, ``hf``,
  ``dispatch``, ``subaperture``).

- **`lumenairy.raytrace`** -- geometric ray tracing
  (``core`` -- the previous ``raytrace.py``).

- **`lumenairy.elements`** -- optical-element family
  (``lenses``, ``doe``, ``coatings``, ``freeform``, ``elements``,
  ``rcwa``, ``polarization``).

- **`lumenairy.io`** -- prescription I/O + storage + codegen
  (``prescriptions``, ``hdf5`` -- previously ``hdf5_io``,
  ``storage``, ``codegen``).

- **`lumenairy.analysis`** -- analysis & post-processing
  (``analysis``, ``detector``, ``ghost``, ``interferometry``,
  ``phase_retrieval``, ``coherence``, ``through_focus``,
  ``plotting``).

- **`lumenairy.sources`** -- source-field generators
  (``core`` -- the previous ``sources.py``).

- **`lumenairy.optimize`** -- prescription optimization
  (``core`` -- the previous ``optimize.py``, ``multiconfig``).

Backwards-compatibility shims are in place at every previously
top-level location: ``from lumenairy.propagation import X``,
``from lumenairy.lenses import Y``, ``from lumenairy.raytrace
import Z``, etc. all continue to work via thin re-export modules
that mirror the moved submodule's namespace (including private
names that some downstream code reaches into).

Each subpackage's ``__init__.py`` mirrors all of its submodules'
namespaces, so ``from lumenairy.elements import apply_real_lens``
works as a flat alternative to ``from lumenairy.elements.lenses
import apply_real_lens``.

### Multi-backend infrastructure

- **`lumenairy.backend.array`** -- `array_namespace(*arrays)`
  returns the appropriate numpy / cupy / jax.numpy namespace.
  Mixing arrays from different backends raises ``TypeError``.

- **`lumenairy.backend.fft`** -- public 2-D / 1-D FFT entry points.
  Preserves the long-standing pyFFTW > scipy.fft > numpy.fft
  priority chain (with plan caching, bad-shape blacklist, automatic
  fallback) for NumPy arrays; CuPy arrays use cuFFT; **JAX arrays
  use `jax.numpy.fft` so calls are differentiable via `jax.grad`
  and JIT-compilable via `jax.jit`**.

- **`lumenairy.backend.random`** -- `RandomState` wrapper that
  accepts an integer seed, `np.random.Generator`,
  `cp.random.Generator`, or `jax.random.PRNGKey`.

- **`lumenairy.backend.scipy`** -- compatibility layer for
  scipy.special / scipy.linalg on multi-backend arrays.

### New propagators

- **`lumenairy.propagators.hfpi`** -- Huygens-Fresnel Path
  Integration.  Monte Carlo ray-based diffraction.  Handles
  cascaded diffraction natively.  `PathBundle`,
  `init_paths_from_field`, `propagate_to_plane`,
  `apply_aperture_diffraction`, `accumulate_to_grid`,
  `propagate_hfpi_freespace_aperture`.

- **`lumenairy.propagators.gbd`** -- Gaussian Beamlet Decomposition.
  Deterministic ray-based diffraction (100x faster than HFPI for
  smooth refractive systems).  `BeamletBundle`, beamlet ABCD
  evolution through free space and thin lenses, coherent
  recombination.

- **`lumenairy.propagators.hf`** -- Van-Vleck-corrected
  deterministic Huygens-Fresnel propagator with the density factor
  in the integrand.

- **`lumenairy.propagators.dispatch`** -- top-level smart-method
  `propagate(E, ..., method='auto')`.

- **`lumenairy.propagators.subaperture`** -- patch / subaperture
  decomposition utilities.

### Bundle field-name unification

The new `PathBundle` (HFPI) and `BeamletBundle` (GBD) data
structures share field names ``positions`` / ``directions`` with
the existing `RayBundle` so user code can operate on any bundle
type with a uniform vocabulary.

### Optional dependencies

- **`jax`** -- JAX for CPU (autodiff + XLA JIT).
- **`jax-gpu`** -- JAX with CUDA wheels.

### Documentation

- **`REFERENCES.txt`** -- new top-level file consolidating every
  external citation (papers, dissertations, standards) the codebase
  draws on.  Inline citations have been removed from source
  comments and docstrings; this file is the single point of
  reference.

### Naming convention

- The documented import alias is now **``import lumenairy as la``**
  throughout the README, CHANGELOG, and all tests / examples
  (previously ``as op``).

### Testing

- 6 new validation files (`test_backend`, `test_hfpi`, `test_gbd`,
  `test_hf`, `test_dispatch`, `test_subaperture`) covering 46 new
  tests, all passing.
- All 23 existing test files continue to pass with no regressions:
  full suite is **23/23 files green**, all individual tests
  passing.

## [3.3.3] — 2026-05-05

### Feature — `recommend_grid_for_prescription`

Design-time companion to `check_grid_vs_apertures` that *recommends*
a grid (`N`, `dx`) instead of just *checking* one.  Given a
prescription, wavelength, source waist, and (optionally) the DOE
order range / period / DOE-to-destination distance, returns the
required half-extent and a sampling pitch that keeps the source
Gaussian, every surface aperture, and every DOE diffraction order
inside the grid with margin.  Optionally rounds `N` to the next
power of two for FFT-friendly sizing.

```python
rec = la.recommend_grid_for_prescription(
    prescription, wavelength,
    source_waist=120e-6,
    doe_orders_max=8,
    doe_period=2.5e-6,
    doe_to_destination_distance=300e-3,
)
N, dx = rec['N'], rec['dx']
```

The output round-trips with `check_grid_vs_apertures` (no warnings
fired at the recommended grid).

### Feature — `scale_prescription`

Geometric self-similarity utility that scales a prescription by a
single factor, preserving F-number, NA, and magnification.  Scales
aperture diameter, semi-diameters, object distance, all thicknesses,
all radii (including biconic Y), aspheric coefficients
(`A_n / factor**(n-1)` so `A_n * r**n` is invariant under
`r -> r * factor`), and coordinate-break decenters/thicknesses.
Does *not* scale conics, glass identities, tilts, or wavelength
(those are scale-free).  Useful for swapping between mm-scale and
m-scale designs without re-deriving every surface field.

```python
big = la.scale_prescription(small, factor=10.0)  # 10x linear
```

### Feature — Endpoint-anchored Chebyshev nodes for `fit_canonical_polynomials`

New `endpoint_anchored=False` kwarg on `fit_canonical_polynomials`.
When `True`, the Chebyshev-Gauss roots are rescaled so the outermost
node sits exactly on the [-1, 1] boundary.  This gives lower max
error for fits whose support is the full source / pupil box (vs.
the standard Gauss roots, which leave a gap to the edge).  Defaults
to `False` to preserve existing fit numerics.

### Docs — `apply_real_lens_maslov` integration modes

Expanded the `integration_method` docstring to fully document all
three modes — `'quadrature'` (current default), `'local_quadrature'`
(per-pixel v2-disk via Newton + Hessian; more rigorous than a global
linear fit at the cost of one Newton solve per output pixel), and
`'stationary_phase'` (zeroth-order saddle).  The library always had
all three; the discoverability of `'local_quadrature'` was poor.

## [3.3.2] — 2026-05-04

### Feature — Embedded grating diffraction in `trace()` and `fit_canonical_polynomials`

`trace()` and `fit_canonical_polynomials` gain a new `surface_diffraction`
keyword argument that pins a chosen DOE / grating order at a specific
surface inside the prescription.  This unblocks LG-aberration-tensor
analysis (and the asymptotic propagator) at non-zero diffraction
orders -- previously, geometric tracing only saw the (0, 0) order
because `apply_doe_phase_traced` operates on standalone `RayBundle`
objects, not surfaces in a sequential prescription.

```python
fit = la.fit_canonical_polynomials(
    prescription, wavelength,
    source_box_half=...,
    surface_diffraction={
        doe_surf_idx: (m_x, m_y, period_x, period_y),
    },
)
```

The kick obeys the standard grating equation
`L_new = L + m_x * lambda / period_x` (and same on y) at the
specified surface, applied AFTER refraction.  Evanescent orders
(`L_new**2 + M_new**2 > 1`) flag rays `alive=False` with
`error_code=RAY_EVANESCENT`.

**Importantly, the OPL accumulator IS updated** with the grating's
linear phase contribution `m * lambda * (x, y) / period` evaluated at
the ray's DOE-plane intersection -- the "constant phase shift"
`apply_doe_phase_traced` explicitly does NOT add but the LG
aberration fit needs to see in order to give correct (0, 0)-piston
phases per emitter.  Without this, the per-emitter pistons at a
non-zero order are inconsistent with the fit's chief-ray landing,
and the LG aberration tensor's piston channel reports nonsensical
inter-emitter phase relationships.

3 new tests in `validation/test_raytrace.py` cover the angular kick,
the OPL contribution, and evanescent-order flagging.

## [3.3.1] — 2026-05-02

### Feature — Pre-flight grid vs prescription-aperture check

`apply_real_lens`, `apply_real_lens_traced`, and `apply_real_lens_maslov`
now run a one-shot check at entry that compares each surface's
`semi_diameter` against the simulation grid's half-extent (`N*dx/2`)
and emits a `UserWarning` if any surface exceeds the grid.

This is the silent-energy-loss case where the lens itself would have
transmitted energy past `N*dx/2` but the simulation grid's hard
boundary clips it.  It manifests downstream as a uniform inward
centroid bias and missing power, and is otherwise difficult to
distinguish from real aberration.  The warning lists the offending
surfaces with their semi-diameters and the largest gap, and points
the user to either grow `N` or coarsen `dx`.

**New public API:**

- **`check_grid_vs_apertures(prescription, N, dx, *, safety_factor=1.0)`**.
  Returns a list of `(label, semi_aperture_m, grid_semi_m, gap_m)`
  for every prescription surface whose `semi_diameter` exceeds
  `safety_factor * N * dx / 2`.  Empty list means the grid is wide
  enough.  Pass `safety_factor=0.95` to flag surfaces that come
  within 5% of the grid edge (recommended for clean Gaussian-wing
  containment).

The warning fires once per call site (Python's default warning
filter dedups by source line), so heavy multi-element systems do
not get spammed.

### Feature — Quadoa Optikos `.qos` import/export (best-effort)

`export_quadoa_qos` / `load_quadoa_qos` add round-trip support for a
Quadoa-Optikos-style JSON system file.  Quadoa's official schema is
not fully publicly documented, so this writer emits a self-defined
JSON layout (schema version `QUADOA_SCHEMA_VERSION = '1.0'`) that
captures every field a lumenairy prescription holds:

- per-surface radii (incl. biconic `radius_y`),
- conics (incl. `conic_y`),
- aspheric coefficients (incl. per-Y axis),
- glasses on both sides of the surface,
- thicknesses, semi-diameters, comments,
- aperture diameter, stop index, wavelength, and units.

Round-trips losslessly inside lumenairy.  External Quadoa
readability is **not yet verified** — for verified interchange,
validate against a known-good reference `.qos`; the docstring
calls this out explicitly.

The library now has full I/O support for Zemax (`.zmx`, `.txt`),
Code V (`.seq`), and Quadoa Optikos (`.qos`).

Validation: 4 new tests in `validation/test_io.py` covering doublet
round-trip, `units='MM'` round-trip, asphere coefficients +
semi_diameter + biconic Y round-trip, and a sanity check that a
round-tripped prescription drives `apply_real_lens` without error.

## [3.3.0] — 2026-05-03

### Feature — Phase-space asymptotic propagator and Laguerre-Gaussian aberration tensor

A new module `lumenairy.asymptotic` implementing the closed-form
Gaussian-moment evaluation of the phase-space (Maslov) diffraction
integral.  This complements
the existing `apply_real_lens_maslov` -- which evaluates the same
underlying integral by direct Chebyshev-quadrature in v_2 -- by
replacing the quadrature with a finite Wick-contracted moment over a
complex-symmetric covariance matrix built from the Chebyshev
polynomial fit.

**What's new:**

- **`fit_canonical_polynomials(prescription, wavelength, ...)` ->
  `CanonicalPolyFit`**.  Trace a 4-D Chebyshev-node grid through any
  prescription, fit Phi(s2, v2) and s1(s2, v2) as 4-variable
  Chebyshev tensor-product polynomials, and return a fit container
  with analytic gradient evaluation.  Sub-microwave residual on
  refractive systems; includes a linear-phase-extraction step that
  restores Nyquist sampling for diffractive surfaces at non-zero
  orders.

- **`aberration_tensor(fit, s2_image, ...)` -> `AberrationTensorResult`**.
  Compute the Laguerre-Gaussian aberration tensor T_{k;n,m} at a
  chief-ray image point.  Indices (p, ell) of the output basis
  correspond directly to classical Seidel/Zernike aberrations:
  (0, 0) is piston/Strehl, (1, 0) is defocus, (2, 0) is primary
  spherical, (1, +-1) is coma, (0, +-2) is astigmatism, (0, +-3)
  is trefoil, etc.  Closed-form Wick-contracted Gaussian moment;
  no quadrature.

- **`propagate_modal_asymptotic(fit, source_amplitudes,
  pupil_amplitudes, ...)` -> ndarray**.  Closed-form leading-order
  asymptotic propagator on a 2-D output grid.  Reduces to Collins'
  ABCD law in the source-dominated limit (large source waist) and
  to the Fourier-of-pupil diffraction-limited spot in the
  pupil-dominated limit; interpolates smoothly between with no
  special handling of caustics.  ~10**3 to 10**4 times faster per
  pixel than direct quadrature; with NaN guards on Newton
  divergence near caustics or out-of-box pixels.

- **`solve_envelope_stationary(fit, s2, source_point, w_s, w_p, ...)`**.
  Newton-solve the envelope-stationary equation for
  the v_2* that maximises the joint Gaussian envelope.  Used inside
  the propagator and the aberration tensor; exposed for users who
  want to inspect the chief-ray geometry directly.

- **`LGAberrationMerit(targets={(p, ell): weight, ...},
  field_points=[...], ...)`**.  A new `MeritTerm` subclass that
  drops directly into `design_optimize`.  Targets named aberration
  channels (defocus, spherical, coma, ...) by output LG index;
  single-call evaluation via `aberration_tensor`.  No wave leg
  required (`needs_wave = False`), so the merit runs at
  millisecond-per-evaluation cost while measuring the same
  physically-named aberrations the wave leg cares about.

- **LG / HG basis utilities** (`lg_polynomial`, `hg_polynomial`,
  `evaluate_lg_mode`, `evaluate_hg_mode`, `decompose_lg`,
  `decompose_hg`, `lg_seidel_label`).  Polynomial-coefficient
  representation of the Laguerre-Gaussian and Hermite-Gaussian
  bases as Cartesian polynomial * shared Gaussian envelope -- the
  form needed by the closed-form Gaussian-moment integrators.
  Verified orthonormal to machine precision on circular
  (LG, w=1mm) and elliptical (HG, wx=1mm wy=1.5mm) cases.

- **Wick moment utilities** (`gaussian_moment_2d`,
  `gaussian_moment_table_2d`).  Closed-form 2-D Gaussian moment
  evaluator for complex-symmetric covariances, with a moment-table
  builder for amortising across many mode-pair contractions.
  Verified against Isserlis identities and direct numerical
  quadrature.

**Why this matters for design optimisation:**

The wave-leg-aware merits (`StrehlMerit`, `RMSWavefrontMerit`, etc.)
are physically faithful but expensive (full ASM propagation per
evaluation).  The ray-leg-only merits (`SphericalSeidelMerit`,
`FocalLengthMerit`) are cheap but only see paraxial geometry; on
high-NA / strongly-aberrated systems they can drive an optimisation
in directions the wave leg disagrees with.

`LGAberrationMerit` is the missing middle tier:  wave-leg-faithful
quantities (the named aberrations the diffraction integral sees) at
ray-leg-only cost.  It is the recommended primary merit for
diffraction-limited design optimisation that needs to converge
quickly across many parameter sweeps (e.g. radii + thicknesses +
conics + aspherics simultaneously).

**Validation:**

A new test file `validation/test_asymptotic.py` covers all 32
identities and end-to-end paths:

- LG / HG basis orthonormality (round / elliptical waist) to 1e-14.
- Wick moment identities:  unit zeroth moment, second moments
  match Sigma_ij to 1e-12, fourth-moment Isserlis identities,
  hand-computed sixth-moment correctness, closed-form vs.
  numerical quadrature agreement to 1e-15.
- Polynomial multiply, shift, and linear substitution unit tests.
- LG / HG decomposition round-trip recovers a known mode.
- Canonical fit:  sub-microwave Phi residual on N-BK7 singlet,
  round-trip evaluation matches direct ray trace, J = ds1/dv2
  has non-trivial magnitude (catches single-source-point
  degeneracy), in_box mask correctness, linear-phase round-trip.
- Newton stationary solver converges in 1 iteration on a clean
  on-axis singlet test.
- Modal propagator:  finite-valued field, PSF peaks at the
  on-axis chief-ray image point.
- Aberration tensor:  evaluates end-to-end with the right shape
  and finite content.
- LGAberrationMerit:  evaluates without error, responds to
  curvature changes, returns a finite penalty when the prescription
  is degenerate (no exceptions propagated to the optimiser).

> Validation: 32/32 new tests pass.  Full library suite of 17
> existing files re-runs green:  no regressions introduced.

**Compatibility:**

No breaking changes.  All existing APIs unchanged.  New module is
purely additive; new merit term subclasses `MeritTerm` and uses the
same `EvaluationContext` as every other merit.

## [3.2.15] — 2026-05-03

### Feature — `apply_doe_phase_traced`: grating diffraction-order shift for ray bundles

New public function in `lumenairy.raytrace` for splitting a
`RayBundle` into one or more diffraction orders at a thin grating /
DOE plane.  Applies the grating-equation direction-cosine shift
`L_new = L + m_x * lambda / period_x` (and the same on the y-axis)
to every ray, recomputes `N` from the unit-norm constraint, and
flags evanescent orders (`L'^2 + M'^2 > 1`) as `alive=False` with a
new error code `RAY_EVANESCENT = 5`.

Two calling conventions:

- **Scalar orders** -- pass `order_x`, `order_y` as scalars; returns a
  bundle the same length as the input.
- **Order arrays** -- pass 1-D arrays of equal length; returns a
  replicated bundle in *order-major* layout (all rays for order 0,
  then order 1, ...).  This is the form used to split a single
  pre-DOE bundle into N orders for one downstream `trace()` call.

Use case: ray-trace through a Dammann splitter or any thin grating
in a sequential prescription.  Before this, callers had to construct
`RayBundle` instances directly and apply the k-shift inline; this
function packages the bookkeeping (broadcast, evanescent flagging,
`error_code` propagation under the first-failure-wins invariant)
and matches the public `trace` / `make_*` API conventions.

Exports: `apply_doe_phase_traced`, `RAY_EVANESCENT`.

> Validation: all 32 raytrace tests pass (6 new), 17 optimize tests
> pass.

## [3.2.14.1] — 2026-04-25

### Bugfix — H-cache OOM at very large N (regression introduced in 3.2.14)

Mirrors the core 3.2.14.1 fix.  The 3.2.14 ASM transfer-function
`H` cache was bounded by entry count (default 8) but not by
bytes; at N=32768 each H is 16 GB so the cache could hold up to
~128 GB of transfer functions, starving `apply_real_lens` of the
RAM it needs for its own sag intermediates.  Caught running
Design 51 traced simulations at N=32768 -- the run failed with
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate
8.00 GiB ...` deep inside `surface_sag_general` partway through
the second lens group.

The H cache now enforces a **per-entry size cap** (default 2 GB,
silently rejects entries above) and a **total bytes budget**
(default 8 GB, LRU-evicts to fit).  At N=32768 the cache
transparently disables itself; lookups miss, H is rebuilt per
call, the result is still correct.  Tunable via
`set_asm_cache_size(h_max_bytes_per_entry=, h_max_total_bytes=)`.

No GUI-side changes; the GUI inherits the safer cache policy
automatically.

> Validation: all 16 files / 298 assertions pass on both
> libraries.

## [3.2.14] — 2026-04-24

### Performance — ASM caches + multi-slot FFTW + batched JonesField + numba aspherics (mirrors core)

Mirrors the core library's 3.2.14 perf pass.  No GUI changes; UI
library version bumped in lock-step.  See the core CHANGELOG for
the per-feature breakdown.

Highlights for UI users:
- Wave Optics simulations that propagate at the same z multiple
  times (multi-config sweeps, per-wavelength loops, optimization
  iterations) now cache the ASM transfer function H — repeat
  propagations are ~1.55× faster at N=2048.
- `JonesField.propagate` runs Ex/Ey through a single batched FFT
  pair on grids ≥ 512.
- `set_default_complex_dtype(np.complex64)` is now exposed at the
  package top level — flip it once in your wave-optics preset for
  ~1.6× FFT throughput and ~2× memory headroom (all propagators
  preserve the caller's dtype, and the existing kernel-phase
  mod-2π folding keeps accuracy at the float32 noise floor).
- pyFFTW now keeps up to 8 plans resident (multi-slot LRU) so
  switching between (Ex, Ey) shape and a 3-D batch shape no longer
  thrashes the plan cache.
- `apply_real_lens` aspheric loop is JIT-fused via numba — pure
  spheres unaffected, aspherics get a single threaded pass.

> **Total**: 16 files, 298 Harness assertions, all PASS.

## [3.2.13] — 2026-04-24

### Validation — physics & interop hammer expansion (mirrors core)

Roughly +70 new test cases added to the core library's validation
suite covering cross-pipeline interop and physics invariants.  No
GUI changes in this release; UI library version bumped to track the
core in lock-step.  See the core CHANGELOG for the per-file
breakdown.

> **Total**: 16 files, 298 Harness assertions across topic suites
> (74 net new vs. 3.2.9 baseline).  All pass.

## [3.2.12] — 2026-04-24

### UI — full polish pass: keyboard, drag-drop, persistent metrics, REPL, compact mode

A round of quality-of-life enhancements covering navigation,
visibility, customization, and ad-hoc analysis.  Core library
unchanged.

**Quick wins**

- **`Ctrl+1` … `Ctrl+9` jump between workspace tabs.** Match the
  Zemax/optiland muscle memory; no more mouse trips to the tab bar.

- **Window title reflects the loaded file + dirty state.** Format is
  `Optical Designer — file.zmx*`, with `*` appended when the design
  has unsaved changes.  Cleared by Save / Open / New.

- **Drag-and-drop `.zmx`, `.txt`, `.seq`, `.json` onto the window**
  to load.  Uses the same paths as File > Open.

- **Permanent right-aligned status-bar metrics**: EFL, BFL, f/#, EPD,
  wavelength.  Visible on every workspace, no need to keep System
  Data open just to glance at headline numbers.

**Workspace upgrades**

- **Pinned docks across all workspaces.**  New
  `View > Workspace > Pin Docks Across All Workspaces…` dialog lets
  you mark docks that should be visible on every tab — handy for the
  Element Table or System Data dock you always want at hand.
  Pinning state persists in `QSettings`.

- **Workspace export/import.**  `View > Workspace > Export
  Workspaces to File…` writes the full workspace set (titles, dock
  membership, saved geometry, pinned set) as a JSON `.workspace`
  file.  Import restores everything in one go — share custom layouts
  with collaborators.

- **`defaults_revision` migration.**  Saved-blob migration appends
  any new default workspace to existing user setups, so previously-
  shipped users automatically get the new `Optimize` tab from 3.2.11
  the next time they launch.  Customizations are preserved.

- **Optimizer progress badge.**  While the optimizer runs, the
  *Optimize* tab title becomes `Optimize • iter N` and the status
  bar reports merit; both clear when finished.  Lets you stay on
  Analysis or Wave Optics while the optimizer runs.

**New docks**

- **`Welcome` (`ui/welcome_dock.py`).** Empty-state landing panel
  with quick-start buttons (Open Demo, Insert Singlet, Insert
  Achromat, Browse Library, Keyboard Shortcuts) and a recent-files
  list backed by `QSettings`.  Default in the Design workspace; auto-
  populates from your last 10 opens / saves.

- **`Python` (`ui/repl_dock.py`).** Embedded Python REPL with
  `model`, `np`, `plt`, `result`, `wave` pre-bound to the live
  system, latest geometric trace, and latest wave-optics result.
  Up/Down arrow history, expression-vs-statement detection (echoes
  values like the standard REPL), captured stdout/stderr.

**Element-table polish (`ui/element_table.py`)**

- **Right-click context menu** on element rows: Duplicate, Delete,
  Move Up/Down, Toggle Distance Variable.  Endpoint rows (Source /
  Detector) and Source-distance variable are correctly disabled.

- **Variable highlighting.** The `Elem#` cell turns amber when the
  element has any optimization variable on it; the `Distance` cell
  turns amber when *distance* itself is a variable.  Quick visual
  for which surfaces the optimizer is allowed to touch.

- **Search box** in the toolbar (`Search elements…`).  Hides rows
  whose Name doesn't contain the substring (case-insensitive).

**Other**

- **F11 / View > Compact Mode.**  Hides the menu bar and replaces
  every dock's title bar with an empty widget — maximises working
  area for laptop screens.  F11 toggles back; the workspace tab bar
  stays visible the whole time.

- **`closeEvent` saves recent files** alongside workspace state.
  `File > New` now resets path + dirty so the title goes back to
  plain "Optical Designer".

> Saved layouts from 3.2.10 / 3.2.11 are auto-migrated: missing
> default workspaces (Welcome, Optimize, etc.) are appended without
> overwriting your customizations.  If you want a clean slate use
> `View > Workspace > Reset Workspaces to Defaults`.

## [3.2.11] — 2026-04-24

### UI — default workspace tweaks: Optimize tab + leaner Design / Wave Optics

Refined the default tab membership in response to a still-too-crowded
Design tab.  Core library and workspace machinery unchanged.

- **New `Optimize` tab** between Design and Analysis.  Holds Optimizer,
  Sliders, Multi-Config, Snapshots, and 2D Layout + System Data for
  context.  These docks were pulled out of the Design tab so the
  optimization workflow lives in its own focused space.

- **Leaner `Design` tab.** Now just 2D Layout, 3D Layout, System Data,
  and Library — the four docks you actually look at while *building*
  the optical layout.  Optimizer/Sliders/Multi-Config/Snapshots moved
  to Optimize.

- **Jones Pupil dropped from `Wave Optics` defaults.**  Still
  available via View > Jones Pupil or by adding it through Manage
  Docks; just no longer shown by default since it is a specialized
  polarization tool not needed for most wave-optics work.

> Existing users with a saved layout from 3.2.10 will continue to
> see their old defaults until they pick *View > Workspace > Reset
> Workspaces to Defaults*.

## [3.2.10] — 2026-04-24

### UI — top-of-window workspace tabs grouping docks by topic

Reduced GUI clutter by introducing a tabbed-workspace system at the top
of the main window.  Each tab shows only the docks relevant to that
phase of design work; the user can create and customize their own.
Core library unchanged.

- **`ui/workspace.py`** — new module with:
  - `Workspace` — named layout (title, dock_names list, saved
    `QMainWindow.saveState()` blob).
  - `WorkspaceBar` — top-of-window QToolBar containing a `QTabBar`
    plus a `＋` button.  Right-click any tab for Manage Docks /
    Rename / Duplicate / Delete; double-click to rename.
  - `ManageWorkspaceDialog` — checkbox list of every dock, with All /
    None bulk toggles, for picking which docks belong to a workspace.
  - `WorkspaceManager` — owns the workspace list and the "current"
    index; `apply_index(i)` switches tabs by hiding non-member docks
    and restoring the per-tab `restoreState()` blob; tracks user-
    initiated dock visibility changes (close button + View menu) and
    updates the active workspace's dock_names so the membership
    sticks; serializes to JSON for `QSettings` persistence.

- **Default workspaces** (loaded on first run, restored thereafter):
  - **Design** — 2D/3D Layout, System Data, Library, Multi-Config,
    Optimizer, Sliders, Snapshots.
  - **Analysis** — 2D Layout, Spot, Ray Fan / OPD, Footprint,
    Distortion, Spot vs Field, Through-focus, PSF/MTF, Field
    Browser, System Data.
  - **Wave Optics** — 2D Layout, Wave Optics, Zernike,
    Interferometry, Phase Retrieval, Jones Pupil, Ghost.
  - **Tolerancing** — 2D Layout, Tolerance, Sensitivity, System Data.
  - **Materials** — Materials, Glass Map, Library, System Data.

- **`ui/main_window.py`** — wired the workspace system into the shell:
  - Added `_build_workspace_bar()` that places the tab strip in the
    top toolbar area with `addToolBarBreak` underneath, so the main
    toolbar lands on the row below the tabs.
  - Added `_init_workspaces()` that builds the dock registry from
    `findChildren(QDockWidget)`, restores from `QSettings` if
    available (else loads defaults), wires every dock's
    `toggleViewAction().toggled` so user toggles update the current
    workspace's dock list, and applies the active workspace.
  - Added handlers for add / rename / duplicate / delete / manage,
    plus a `View > Workspace` submenu (with Reset to Defaults).
  - Added `closeEvent` to flush the current layout and persist all
    workspaces to `QSettings('lumenairy', 'OpticalDesigner')` so
    custom workspaces survive restart.
  - Per-tab dock geometry preserved: `save_current_layout()` snapshots
    `saveState()` into the outgoing workspace before each switch, so
    drags / resizes within a workspace are not clobbered.

### Why

The main window was getting too crowded — at 3.2.9 we had 27 dock
widgets stacked into 3 dock-area tab groups.  Tabbed workspaces let
the user focus on one phase at a time (designing the layout, then
analyzing it, then doing wave optics) without losing access to any
dock — and they can build their own analysis tabs ("MTF only",
"Distortion only", etc.) for whatever they want to plot.

## [3.2.9] — 2026-04-24

### UI — three new analysis docks + command palette + system-data additions

Filled out the comparison vs optiland's GUI feature set.  Core
library unchanged.

- **`ui/footprint_dock.py`** — per-surface ray-bundle outline.  For
  every surface in the system, plots `(x, y)` of the alive rays from
  a `make_rings(rings, per_ring)` launch with the clear-aperture
  circle drawn as a reference.  Multi-field overlay (configurable).
  Standard tool for verifying surface diameters, stop placement, and
  vignetting at every interface, not just the image plane.

- **`ui/distortion_dock.py`** — chief-ray distortion vs field +
  distortion grid.  Sweeps field angles 0..max, traces the chief
  ray, plots `100·(h_chief - f·tan θ) / (f·tan θ)` vs field, and
  also draws a reference paraxial grid (red) overlaid on the actual
  image-plane chief-ray positions (blue).  Status line reports max
  distortion + Pincushion/Barrel tag.

- **`ui/spot_field_dock.py`** — N×M array of spot diagrams across
  the configured `model.field_angles_deg`, on a shared scale so
  cross-field aberration growth is visible at a glance.  Optional
  Airy-disc overlay; per-panel RMS in titles; configurable
  rings / per-ring.

- **`ui/command_palette.py` (Ctrl+K / Ctrl+Shift+P)** — VS-Code-style
  fuzzy-search dialog over every menu action.  Indexes the live
  `QMenuBar` at popup time so any menu action labelled "Foo > Bar
  > Baz" is reachable by typing "fbb" (or "baz", or "f bar").
  Character-subsequence fuzzy match with word-boundary boost and
  prefix-match priority.  Up/Down navigation, Enter to fire, Esc
  to dismiss.  Hooked into MainWindow via `install_command_palette`
  in `__init__`; also added under `Help > Command Palette...` for
  discoverability.

### UI — extended SystemSummary

- `SystemSummaryWidget` now also reports:
  - Multiple wavelengths (when more than one is configured)
  - Configured field angles + total FOV
  - Working f/# (= image-space f/# for object at infinity)
  - Image-space NA
  - Airy disc radius
  - Front and rear principal planes (Welford convention)
  - Stop surface index (via `find_stop`)
  - Paraxial entrance/exit pupil positions and radii (via
    `compute_pupils`, when defined for the system)

### Notes

- 33 UI modules import cleanly (was 29 in 3.2.8; +3 new docks +
  command_palette).
- Footprint, Distortion, and Spot-vs-Field docks all run end-to-end
  against the AC254-100-C demo doublet:
  - Footprint draws S1 / S2 / S3 / Image with vignetting visible.
  - Distortion reports 0.0277 % Barrel at ±5° on AC254-100-C.
  - Spot-vs-Field shows 0°/1°/2° panels with RMS 41.5 / 39.0 /
    49.0 µm and the 5.29 µm Airy disc overlay.
- Command palette fuzzy match verified for "psf", "ghost",
  "arcoat", "field" against the indexed menu actions.
- Tier 3.1 (3D rays in Layout3DView) was already implemented by the
  existing `_draw_rays` method (line 235 of `layout_3d.py`); no
  duplication added.
- Core regression 16/16 (251 assertions) still passes on both
  libraries.

## [3.2.8] — 2026-04-24

### UI — Lens-function options dialog

- **New top-level `&Options` menu** (between Preferences and Help)
  with a `Lens function options...` entry that opens a tabbed
  dialog for configuring kwargs of the three real-lens pipelines.

- **New `ui/lens_options_dialog.py`** (`LensOptionsDialog`).
  QTabWidget with one tab per function — `apply_real_lens` (7
  kwargs), `apply_real_lens_traced` (11 kwargs),
  `apply_real_lens_maslov` (9 kwargs).  Widgets are built
  procedurally from a single `LENS_KWARG_REGISTRY` mapping; adding
  a new kwarg to a function is a one-line registry entry.  Each
  field has a tooltip explaining what it does.

- **Widget kinds**: bool → QCheckBox, int → QSpinBox (with min/
  max/step), float → QDoubleSpinBox, enum → QComboBox.  "Reset
  this tab" and "Reset all to defaults" buttons clear overrides.

- **Default filtering**: only kwargs whose value differs from the
  library default are persisted on `model.lens_options`.  Keeps
  the stored state minimal and means an omitted kwarg gets the
  current library default automatically — useful when the library
  changes a default and the user doesn't want to track it.

- **`SystemModel.lens_options`** — new dict-of-dicts attribute
  (`{function_name: {kwarg: value}}`) holding the user's overrides.
  Initialised empty in `__init__`.

- **WaveOpticsDock integration** — when delegating to a real-lens
  function, the worker reads `model.lens_options[func_name]` and
  splats it onto the call.  Dock-level controls (`ray_subsample`,
  `tilt_aware_rays`) remain authoritative; the dialog only
  contributes a value when the dock didn't already.

### Notable kwargs newly exposed

  apply_real_lens:
    bandlimit, fresnel, absorption, slant_correction,
    seidel_correction, seidel_poly_order, wave_propagator

  apply_real_lens_traced:
    bandlimit, ray_subsample, preserve_input_phase,
    tilt_aware_rays, fast_analytic_phase, parallel_amp,
    inversion_method, newton_fit, newton_poly_order,
    on_undersample, wave_propagator

  apply_real_lens_maslov:
    integration_method, poly_order, n_v2, ray_field_samples,
    ray_pupil_samples, extract_linear_phase, collimated_input,
    output_subsample, normalize_output

### Verification

- 29 UI modules import cleanly (was 28; +1 new dialog file).
- Dialog instantiates with 3 tabs, 27 kwargs total.
- Default-filtering verified: only changed values persist.
- Reset-this-tab and Reset-all-to-defaults rebuild widgets in place.
- End-to-end: enabling `fresnel=True` via the dialog produces a
  measured ~8% power loss through a BK7 singlet at 1.31 µm —
  confirming the kwarg flows correctly from dialog → model →
  WaveOpticsDock → `apply_real_lens`.
- Core regression 16/16 (251 assertions) on both libraries.

## [3.2.7] — 2026-04-24

### UI — hardware-self-calibrating forecast model

- The Wave-Optics dock's `forecast_resources` previously hardcoded
  a single "12 ms ASM at N=1024" reference, which over-predicted on
  fast workstations and dramatically under-predicted on laptops or
  slow sandboxes (16× under on a 192-ms-ASM box).  It now
  **self-calibrates against the local CPU** on first use:

  * **`_local_asm_baseline_ms()`** — runs one warmup + two timed
    `angular_spectrum_propagate` calls at N=512 (~50-300 ms total),
    extrapolates to the N=1024 reference via the standard
    ``N² · log N`` cost model, and caches the result for the rest
    of the session.

  * Every CPU-bound coefficient in the time model (Newton per-pixel,
    setup costs) is now multiplied by a `hw_scale = local / 12`
    factor, so the entire forecast scales **linearly** with the
    measured baseline.  A 4-ms-ASM workstation gets ~3× shorter
    forecasts than the 12-ms reference; a 100-ms laptop gets ~8×
    longer.  Validated: forecast for a 3-surface doublet sub=8 at
    N=1024 reads 78 ms / 234 ms / 1.9 s for ASM-1024 baselines of
    4 / 12 / 100 ms respectively — perfectly proportional.

  * **Recalibrate button** added to the Wave-Optics dock just above
    the forecast strip.  Shows the current baseline ("ASM-1024 =
    14.2 ms (self-measured)") and force-re-measures on click.
    Useful after switching FFT backend (NumPy → SciPy → pyFFTW →
    CuPy) or after moving the process to a different machine via
    hibernate / VM migration.  Disables the button while the (sub-
    300 ms) measurement runs so a double-click can't kick off two
    timed propagations at once.

  * **Fallback**: if `lumenairy.propagation` can't be
    imported (broken CuPy install, etc.), the calibration falls
    back silently to the historical 12 ms reference rather than
    crash the dock.

### Notes

- Pure additive change to the UI subpackage.  Core library is
  unchanged from 3.2.6 (Maslov `_ne` fix + recalibrated multipliers).
- Auto-calibration cost is one-time per process: ~50-300 ms on the
  first `_update_forecast` call.  Subsequent forecasts hit the
  cache in microseconds.
- All 28 UI modules still import cleanly; core regression
  16/16 (251 assertions) passes on both libraries.

## [3.2.6] — 2026-04-24

### Fixed — core library

- **`apply_real_lens_maslov` — `NameError: name 'ne' is not defined`**
  in the 3.2.2 Maslov→lenses merge.  The Maslov section uses
  ``numexpr`` which the rest of ``lenses.py`` imports as ``_ne``
  (with an underscore); the merged code's bare ``ne.evaluate(...)``
  references were never rewritten.  Blocked every Maslov call from
  running at all.  Renamed the four ``ne.evaluate`` sites inside
  ``_integrate_quadrature`` to ``_ne.evaluate``.  Regression caught
  by the benchmark sweep (below).

### UI

- **``forecast_resources`` recalibrated** for the perf work that has
  landed across 3.1.3–3.2.2: numexpr-fused phase screens,
  pre-resolved glass indices, polynomial-Newton default, parallel
  amp+amp(pw) pass, amplitude-masked Newton.  New ratios to ASM
  (benchmarked against an AC254-100-C doublet at N=1024, April 2026):

      ASM                  1.0  (reference)
      Fresnel              0.8  (was 1.3 — Fresnel is actually faster
                                 than ASM, no bandlimit kernel)
      Fraunhofer           0.6
      Rayleigh-Sommerfeld  3.3  (was 2.8)
      SAS                  5.0  (new; 3 FFTs at 2N-padded grid)

  Added `real_lens_maslov` branch (~600× ASM on defaults; dominated
  by 2-D quadrature integration).  Replaced the stale "6 FFTs per
  surface" model for `apply_real_lens` with the physically-correct
  ``(n_surfaces - 1)`` ASM-through-glass calls plus a small
  phase-screen overhead.  Replaced the pre-3.1.7 spline-Newton
  constant (``0.8e-6`` s/px) with a polynomial-Newton constant
  (``6e-6`` s/px); net effect is that traced-sub=8 (the new default)
  forecasts at ~230 ms vs ~980 ms previously, matching the ~261 ms
  actual.  Forecasts now within ~2× of measured wall-clock time
  across all code paths (was 10-20× overestimate on some paths).

- **Fixed I/O forecast bug**: a ``max(n_save_planes, 1)`` clamp was
  adding ~800 ms of phantom disk-save time even when the caller
  specified zero planes.  Changed to gated ``if n_save_planes > 0``
  so forecasts for no-save runs are now correct.

### Notes

- Core regression (16 test files, 251 assertions) still passes on
  both libraries after the Maslov fix.
- UI smoke test: 28 UI modules import cleanly; `WaveOpticsDock`
  instantiates with 4 lens-model options (including Maslov) and 5
  propagator options (including SAS).

## [3.2.5] — 2026-04-24

### UI — Tier 1 feature additions

Core library unchanged.  UI-subpackage changes only (UI-variant library).

- **Ghost-analysis dock** (new ``ui/ghost_dock.py``) — one-click
  enumeration of all ordered surface pairs with bare-Fresnel ghost
  intensities ``R_i * R_j``, rendered as a sortable table.  Registered
  in the Bottom dock area and wired to the View menu and
  Analysis > "Ghost analysis".
- **Jones-pupil visualization dock** (new ``ui/jones_pupil_dock.py``)
  — probes the current lens at configurable N/dx with pure-x and
  pure-y plane-wave inputs, renders the canonical 2x4 Jones pupil
  (amplitude + phase for Jxx/Jxy/Jyx/Jyy).  Registered in the Right
  dock area with a View-menu toggle and Analysis > "Jones pupil"
  shortcut.  Verified: scalar demo doublet gives an exactly diagonal
  pupil (``|Jxy| = |Jyx| = 0``).
- **Codegen "Export Python Sim Script..."** menu item (File menu,
  already wired in the codebase) confirmed functional end-to-end
  after fixing the ``SystemModel.to_prescription`` bug described
  below — one-click reproducibility for the current system.

### Fixed

- **``SystemModel.to_prescription`` indentation bug** in
  ``ui/model.py``.  A pair of module-level helper functions
  (``_nice_dx``, ``_next_nice_N``) had been dedented out of the
  ``SystemModel`` class body, which silently pulled
  ``to_prescription`` out of the class as well (Python parsed it
  as dead code inside ``_next_nice_N``).  ``MainWindow``'s File >
  Export menu items (Zemax ZMX / CODE V SEQ / Python Sim Script,
  all of which call ``self.model.to_prescription()``) would have
  failed at runtime.  Moved ``to_prescription`` above the
  module-level helpers so it's a genuine method again.

### Verification

- 28 UI modules still import cleanly (was 26 pre-3.2.5; up 2 from
  the new ghost + Jones-pupil docks).
- 223 imported symbols resolve (was 216).
- ``MainWindow`` builds all docks without segfault in the offscreen
  headless harness (except the VTK 3D renderer path, which is an
  OpenGL/platform issue unrelated to this release).
- Both new docks run end-to-end against the ``AC254-100-C`` demo:
  Ghost dock produces 3 rows; Jones pupil dock returns the
  expected diagonal ``|Jxx| = |Jyy| = 1.32``, ``|Jxy| = 0``.
- Full core validation suite still 16/16 (251 assertions) on both
  libraries.

## [3.2.4] — 2026-04-24

### UI — compatibility audit + new-feature exposure

Core library unchanged in this release; all changes are in the
`lumenairy.ui` subpackage of the UI-variant library.

- **Fixed broken import** in `ui/materials_dock.py`: was importing
  `lumenairy.ui.glassmap_dock` (no underscore) but the
  actual module is `glass_map_dock`.  The tab loaded silently
  via the try/except fallback; glass-map tab now appears again.

- **SAS propagator exposed in `waveoptics_dock.py`** — added `'SAS'`
  to the "free-space propagator between elements" dropdown with an
  auto-resample-back-to-dx dispatch mirroring the existing Fresnel
  handler.  Covered in both the between-surfaces loop and the
  propagate-to-focus block.  Forecast-resources time model updated
  with a conservative 2× multiplier (SAS = 3 FFTs + resample).

- **Maslov lens model exposed in `waveoptics_dock.py`** — added
  `'apply_real_lens_maslov (phase-space, caustic-safe)'` to the
  lens-model dropdown alongside the existing ASM / apply_real_lens
  / apply_real_lens_traced options.  Dispatch routes through the
  lens-router branch; tooltip explains the phase-space /
  stationary-phase rationale.

- **CODE V `.seq` file I/O exposed in `main_window.py`** —
  added `.seq` to the File > Open dialog's filter (sibling of
  `.zmx` / `.txt`), the File > Export Prescription dialog
  (sibling of `.json` / `.zmx`), and the CLI file-load path.
  Dispatches to :func:`export_codev_seq` / :func:`load_codev_seq`.

### Verification

- All 26 UI modules import cleanly.
- 216 imported symbols resolve (was 208 before the UI update
  added 8 new imports for the three features above).
- `waveoptics_dock.py`, `main_window.py`, `materials_dock.py`
  parse cleanly with AST.
- Full core validation suite (16 files, 251 assertions) still
  passes on both libraries.

## [3.2.3] — 2026-04-24

### Added

- **``wave_propagator='fresnel'`` and ``wave_propagator='rayleigh_sommerfeld'``**
  options for :func:`apply_real_lens` (and threaded through
  :func:`apply_real_lens_traced`).  Rounds out the through-glass
  propagator switch to all four physically-sensible choices:
  ``'asm'`` (default), ``'sas'``, ``'fresnel'``,
  ``'rayleigh_sommerfeld'``.  Each follows the same resample-back-to-dx
  pattern (Fresnel, SAS) or preserves the input pitch natively (ASM,
  R-S).  RS was verified to match ASM to ~1e-13 at mm-scale
  through-glass distances as expected.  Unknown values now raise
  ``ValueError`` with a list of supported options.
  4 new assertions in ``test_lenses.py``.

## [3.2.2] — 2026-04-24

### Changed — Maslov propagator merged into lenses module

- The former top-level ``lens_maslov.py`` has been deleted.  Its sole
  public function ``apply_real_lens_maslov`` now lives in
  :mod:`lumenairy.lenses` alongside ``apply_real_lens`` and
  ``apply_real_lens_traced``.  This matches the fact that it is a
  third real-lens wave-optics pipeline (phase-space / Maslov), not a
  separate subsystem.
- Public API unchanged: ``lumenairy.apply_real_lens_maslov``
  and ``from lumenairy.lenses import apply_real_lens_maslov``
  both work.  The legacy path
  ``from lumenairy.lens_maslov import apply_real_lens_maslov``
  is the only thing that breaks; nothing in the library or its
  validation suite was using it.
- All 251 validation assertions pass; 272 public symbols still
  resolve through the connectivity audit.

## [3.2.1] — 2026-04-24

### Added — SAS integration hooks

Three places where the library's built-in propagation path was
hard-wired to ASM now accept SAS as a first-class alternative.

- **`propagate_through_system(method='sas')`**.  On `'propagate'`
  elements, setting `method='sas'` (globally or per-element) routes
  through :func:`scalable_angular_spectrum_propagate` instead of ASM.
  The pipeline auto-resamples the SAS output back to the original
  `dx` between elements so downstream lenses / apertures keep their
  physical coordinates.  Extra per-element keys: ``pad`` (default 2),
  ``skip_final_phase`` (default False).

- **`apply_real_lens(wave_propagator='sas')`** (and forwarded through
  ``apply_real_lens_traced``).  Swaps the through-glass
  ``angular_spectrum_propagate`` call for
  ``scalable_angular_spectrum_propagate`` + resample-back-to-grid.
  Physically ASM remains the appropriate choice inside a lens (glass
  thicknesses are mm-scale, high-Fresnel-number); this switch is
  exposed for research and for pipelines that want a single
  propagator used consistently throughout.

- **`JonesField.sas_propagate(z, wavelength, pad=2, skip_final_phase=False)`**.
  Polarization-aware SAS wrapper that applies the scalar
  :func:`scalable_angular_spectrum_propagate` to ``Ex`` and ``Ey``
  independently (both on the same grid and with the same kernel),
  then updates ``self.dx`` / ``self.dy`` to the new output pitch.
  Requires ``dx == dy`` (square grid); raises ``ValueError`` otherwise.

### Notes

- Additive change; no existing behaviour modified.  All 246
  previously-passing assertions still pass.  New total: 251
  assertions across 16 test files.

## [3.2.0] — 2026-04-24

### Added

- **Scalable Angular Spectrum propagator**
  (``scalable_angular_spectrum_propagate`` in ``propagation.py``).
  Implements the Heintzmann-Loetgering-Wechsler 2023 three-FFT
  kernel: ASM-minus-Fresnel precompensation phase + chirp + FFT.
  Output pitch is ``lambda*z/(pad*N*dx)`` — larger than input at
  long ``z``, avoiding the impractical-N problem of plain ASM.
  Includes paper's closed-form ``z_limit`` check, Fresnel-style
  physical-amplitude prefactor (so power is conserved; the
  reference notebook is amplitude-agnostic), ``skip_final_phase``
  toggle, ``pad`` factor, ``use_gpu`` path, ``verbose`` diagnostics.
  Validated against ``fresnel_propagate`` / ``fraunhofer_propagate``
  in the respective limits.  5 new assertions in
  ``test_propagation.py``.

- **CODE V ``.seq`` import/export**
  (``export_codev_seq`` + ``load_codev_seq`` in
  ``prescriptions.py``).  Round-trips the library prescription dict
  through the canonical CODE V sequence syntax.  Units M/MM/IN.
  4 new assertions in ``test_io.py``.

- **BSDF surface scatter model** (new module ``bsdf.py``) with
  ``LambertianBSDF``, ``GaussianBSDF``, ``HarveyShackBSDF``.
  Common interface: ``evaluate``, ``sample``,
  ``total_integrated_scatter``.  Attached to ``Surface`` via new
  ``bsdf`` field.  Helper ``sample_scatter_rays`` spawns a
  ``RayBundle`` of scattered rays for Monte Carlo stray-light
  propagation.  8 new assertions in ``test_features.py``.

- **Jones pupil spatial-map visualization**
  (``plot_jones_pupil`` + ``compute_jones_pupil`` in
  ``plotting.py``).  ``compute_jones_pupil`` probes a system with
  orthogonal x/y plane-wave inputs and returns the full
  ``(Ny, Nx, 2, 2)`` Jones matrix.  ``plot_jones_pupil`` produces
  the canonical 2x4 grid (amplitude + phase for each matrix
  element) with phase masked below an amplitude threshold.
  4 new assertions in ``test_polarization.py``.

### Notes

- No breaking changes.  All 225 previously-passing assertions
  still pass.  New total: 246 assertions across 16 test files.
- ``Surface`` dataclass gains a new optional ``bsdf`` field.
  ``_surface_copy_with`` propagates it through edits.  Older
  pickled bundles/prescriptions without this field are handled
  transparently via ``getattr(..., None)``.

## [3.1.11] — 2026-04-24

### Added

- **Stop-aware `seidel_coefficients`**.  Added ``stop_index`` and
  ``field_angle`` kwargs.  When the declared stop is not at surface
  0, the chief ray's initial conditions at surface 0 are now derived
  from the pre-stop ABCD so that ``y_chief = 0`` at the stop by
  construction.  Backward-compat: the default behaviour uses
  :func:`find_stop` (which falls back to surface 0 when no surface
  is flagged ``is_stop=True``), matching the pre-3.1.11 assumption
  bit-for-bit.  Output dict now also contains ``'stop_index'`` for
  diagnostics.  ``seidel_prescription`` passes the new kwargs
  through.

  Hopkins/Welford convention + ``H^2``-factored-out sums are
  preserved; chief-ray-dependent coefficients (S2 coma, S3
  astigmatism, S5 distortion) reflect the new stop position
  correctly, while S4 (Petzval, curvature-only) remains invariant
  as expected.

- **`validation/test_validation_lens.py`** — known-answer
  regression test harness covering the major library APIs:
  lensmaker's formula, manual ABCD composition, ``find_lenses``
  auto-detection, stop-aware Seidel invariants, ``compute_pupils``,
  ``refocus`` vs full retrace, ``through_focus_rms``,
  ``apply_real_lens`` vs ``apply_real_lens_traced`` vs
  ``apply_real_lens_maslov``, and per-ray error codes.  Runs
  standalone; exits 0 on all-pass.

### Fixed

- **`refocus` now projects to the requested image plane** instead
  of advancing rays by ``delta_z`` from their current (post-
  refraction) position, which was at ``z = sag`` of the last
  surface rather than at the vertex plane.  The old semantics
  caused off-axis rays to land short of the intended image plane
  by up to ``sag * (1 - 1/cos_angle)`` (~100 um at h=8 mm on an
  F/5 singlet).  New behaviour uses ``(delta_z - z_current)`` in
  the arc-length computation so rays land exactly on ``z =
  delta_z`` in the last surface's frame -- bit-identical to what
  ``trace`` would produce with a flat image plane appended at
  ``thickness=delta_z``.  ``through_focus_rms`` inherits the fix
  automatically.

## [3.1.10] — 2026-04-24

### Added

- **`apply_real_lens(..., use_gpu=False)`** — new kwarg making the
  whole phase-screen + in-glass ASM pipeline array-API polymorphic.
  Default remains ``False`` so CPU output is bit-for-bit backward
  compatible with 3.1.9 (verified: 0.0e+00 max difference on all
  tested inputs).  When ``use_gpu=True`` or ``E_in`` is already a
  CuPy array:

    * Every meshgrid, sag array, phase screen, and per-surface
      multiplication runs natively on device via ``cp.*`` operations.
    * Internal ``angular_spectrum_propagate`` auto-detects the CuPy
      backend (it already had a GPU path) and uses cuFFT for the
      in-glass ASM propagation between surfaces.
    * The numexpr-fused phase-screen path is automatically skipped
      on GPU (numexpr is CPU-only); CuPy's fused elementwise kernels
      handle the ``E * exp(-i k OPD)`` update.
    * The returned array is a CuPy array (not automatically pulled
      back to host) so downstream callers can keep the field on GPU
      for further propagation or masking.  Use ``cp.asnumpy()`` to
      pull back when needed.

- **`apply_real_lens_traced(..., amp_use_gpu=False)`** — new kwarg
  passing ``use_gpu`` through to the internal ``apply_real_lens``
  calls that build the amplitude envelope and the reference phase.
  Default ``False``: no behaviour change unless explicitly enabled.
  When ``amp_use_gpu=True``:

    * The ``amp`` and ``amp(pw)`` passes run on GPU (or just the
      ``amp`` pass when ``fast_analytic_phase=True``).
    * GPU results are pulled back to the host via ``cp.asnumpy()``
      at the end of the amp block, so the ray trace, Newton
      inversion, and final field assembly run unchanged on CPU.
      This lets the existing stable CPU pipeline drive the
      ray-trace side while the FFT-bound amp side offloads.
    * Independent of the Newton-inversion ``use_gpu`` kwarg added
      in 3.1.7: the two GPU flags can be enabled independently.

### Changed

- **`surface_sag_general` / `surface_sag_biconic`** made array-API
  polymorphic.  Detect CuPy vs NumPy from the input array and
  dispatch all internal ops (``zeros_like``, ``where``, ``sqrt``)
  accordingly.  Needed by the GPU ``apply_real_lens`` path; CPU
  callers see no change.

### Performance

- Not measured at production scale on this host (the local CuPy
  install is missing cuSOLVER and cuFFT DLLs, so the GPU path was
  validated for correctness via code-path inspection but couldn't
  be benchmarked end-to-end).  Expected speedup when a complete
  CUDA stack is available: ~5-10x on the amp + amp(pw) passes
  (they're ASM-FFT-bound and cuFFT is substantially faster than
  pyFFTW for large grids), dropping the wall-time contribution of
  those passes from ~50% of ``apply_real_lens_traced`` to ~10-20%.

### Known limitations

- The GPU path requires a complete CuPy install (cuFFT, cuBLAS;
  cuSOLVER only if you separately enable
  ``newton_fit='polynomial'`` with GPU).  Pass ``use_gpu=True``
  explicitly opts in; missing components raise ``ImportError`` at
  the first GPU call.

## [3.1.9] — 2026-04-24

### Added

- **`lens_abcd(lens, wavelength, start=None, end=None, label=None)`**
  — paraxial characterisation of a single lens element.  Accepts
  a prescription dict, a list of ``Surface`` (with optional
  ``start`` / ``end`` slice indices), or a single ``Surface``.
  Strips the trailing-thickness air gap so the returned ABCD is
  air-to-air at the element's own vertex, not "lens plus
  downstream propagation."  Returns a ``LensInfo`` dataclass with
  ``abcd``, ``efl``, ``bfl``, ``ffl``, ``principal_planes``
  (Welford convention), ``thickness``, and surface-index range.

- **`find_lenses(surfaces, wavelength)`** — auto-detect individual
  lens elements in a surface list by scanning air -> glass ->
  air transitions.  Cemented multi-element groups (glass -> glass
  interfaces in the middle) stay grouped.  Mirrors are reported
  as their own one-surface elements.  Returns
  ``List[LensInfo]``.

- **`compute_pupils(surfaces, wavelength, stop_index=None)`** —
  paraxial entrance / exit pupil positions and radii.  Images the
  aperture stop backward (for EP) and forward (for XP) through
  the pre- and post-stop sub-systems using their ABCD matrices;
  no ray tracing.  Returns a ``PupilInfo`` dataclass with
  ``ep_z``, ``ep_radius``, ``xp_z``, ``xp_radius``, and the
  resolved ``stop_index``.

- **Per-ray diagnostic `error_code` on `RayBundle`** — a ``uint8``
  array (1 byte / ray) recording the reason each dead ray was
  killed:

    * ``RAY_OK = 0``             — alive
    * ``RAY_TIR = 1``            — total internal reflection
    * ``RAY_APERTURE = 2``       — clipped by a surface semi-diameter
    * ``RAY_MISSED_SURFACE = 3`` — intersection Newton failed
    * ``RAY_NAN = 4``            — arithmetic produced NaN/Inf

  First-failure-wins: once a ray is killed with a non-zero code,
  subsequent surfaces do not overwrite the root cause.  The
  ``_refract`` and ``_intersect_surface`` helpers now set the
  appropriate code at kill time.  ``trace_summary`` prints the
  breakdown (``[TIR=N, aperture=M, miss=K, nan=L]``).  Bundles
  constructed before 3.1.9 (or without an explicit ``error_code``)
  are handled transparently: ``__post_init__`` synthesises the
  field from ``alive`` as a best-effort placeholder.

### Performance

- **Glass indices pre-resolved once per `trace()`** call via an
  up-front list comprehension, instead of two `get_glass_index`
  lookups per surface inside the hot loop.  Underlying
  ``refractiveindex.info`` cache was already avoiding the
  dispersion calculation; this removes the Python dispatch
  overhead too.  Small per-call win, more useful at high
  repeated-trace counts.

## [3.1.8] — 2026-04-24

### Added

- **`trace(..., output_filter='all' | 'last' | callable)`** — new kwarg
  controlling what per-surface state is retained in
  ``result.ray_history``.  Default ``'all'`` preserves existing
  behaviour.  ``'last'`` keeps only the final image bundle, eliding
  every intermediate ``RayBundle.copy()``.  On large ray counts
  this is a significant memory win:

  - N=32768 `apply_real_lens_traced` call at `ray_subsample=8`:
    ~4M coarse rays × 7 float64 arrays + 1 bool = ~228 MB per
    surface copy.  A 6-surface doublet therefore saves ~1.4 GB
    per call.
  - At larger grids / finer ray sub-sampling the savings scale
    linearly with ray count.

  `apply_real_lens_traced` now calls `trace(..., output_filter='last')`
  internally since it only consumes the image bundle; existing
  callers of `trace` see no change unless they opt in.

- **`refocus(result, delta_z, wavelength=None)`** — closed-form
  image-space transfer of a traced ray bundle.  Advances every
  ray by ``delta_z`` along its direction cosines; updates
  positions and OPL with the correct image-space refractive
  index (reads ``surfaces[-1].glass_after``, not assumed
  ``n=1``).  Signed ``delta_z`` (negative = move toward the
  lens, pre-focus) — the OPL update is ``n * delta_z / N``
  without an absolute-value, so defocus in both directions is
  handled consistently.

- **`through_focus_rms` rewritten on top of `refocus`**: single
  base trace through the real surfaces followed by a
  closed-form transfer at each focus shift, instead of
  rebuilding the surface list and re-tracing from surface 0.
  Expected speedup scales with the number of surfaces (~5-20x
  typical); numerical output matches the pre-3.1.8 full-retrace
  path to RMS spot-size precision (verified on a doublet:
  identical best-focus location, ~1e-4 max RMS difference which
  is finite-ring-sampling noise).

- **`is_stop: bool = False`** field on `Surface`.  Explicitly
  marks the aperture stop when set by a loader or caller.
  Zemax `.zmx` / `.txt` loaders propagate the STOP keyword onto
  the surface via the per-surface ``'is_stop'`` key.

- **`find_stop(surfaces)`** — locate the aperture-stop surface
  index.  Dispatch: first surface with ``is_stop=True`` (warns
  on multiple flags); else first surface with a finite
  ``semi_diameter``; else surface 0 (with a ``UserWarning`` if
  multi-surface).  Foundation for future stop-aware Seidel,
  pupil, and chief-ray aim work.

### Fixed

- **Dead-code line in `_paraxial_trace`** — removed the
  ``if False else u`` clause on the refraction update that was a
  commented-out alternative form of the paraxial refraction
  equation; the active preceding line already implements the
  correct ``u <- u - y * phi / n2`` update.  No observable
  behaviour change (the function was mathematically correct), but
  the expression now states its intent.

- **Seidel ``S4`` (Petzval) double-assignment** in
  ``seidel_coefficients`` — the first of two adjacent
  assignments used ``-c * (n2 - n1) * (1/n2 - 1/n1)``, which
  squares the index difference and is dimensionally wrong; the
  second, which won, used the correct ``-(n2-n1)/(n1*n2) * c``
  form (standard Hopkins/Welford convention with ``H^2``
  factored out).  The errant line is removed.  Output
  ``total['S4']`` is unchanged (the correct line was already
  winning); the cleanup clarifies intent.

### Performance

- `apply_real_lens_traced` at N=32768: ~1.4 GB of transient
  ``RayBundle.copy()`` allocations eliminated per call via
  `output_filter='last'`.  No measurable wall-time regression
  at small N; larger-grid runs should see moderate gains from
  reduced memory pressure and allocator churn.

## [3.1.7] — 2026-04-23

### Added

- **`apply_real_lens_maslov`** (new public function in `lens_maslov.py`).
  Phase-space / Maslov propagator complementing `apply_real_lens` and
  `apply_real_lens_traced`.  Fits a 4-variable Chebyshev tensor-product
  polynomial to the ray-traced back-map `s1(s2, v2)` and `OPD(s2, v2)`,
  then evaluates the Maslov integral by one of three methods selected
  by `integration_method`:

    * `'stationary_phase'` (recommended) — closed-form saddle-point
      evaluation per output pixel.  Caustic-safe by construction, no
      critical-sampling constraint, analytically differentiable w.r.t.
      the polynomial coefficients.  On Design 51 L1 at N=1024
      (collimated Gaussian input, `collimated_input=True`,
      `output_subsample=4`) matches `apply_real_lens_traced` to
      ~1.2 % RMS intensity in 2.2 s — faster than traced.

    * `'quadrature'` — uniform Tukey-windowed Riemann sum on the v2
      grid.  Correct for extended multi-source inputs inside the
      quadrature-validity bound `w_s >= D_s1 / n_v2`; not suitable for
      single-source collimated inputs where the integrand is
      delta-like in v2.

    * `'local_quadrature'` — Hessian-oriented uniform quadrature in a
      small window around the stationary point.  Captures asymptotic
      corrections beyond leading stationary phase.  Extended-source
      regime only.

  Four fast-path speedups are applied throughout:

    1. Precompute the `s2`-basis Chebyshev Vandermonde over the output
       grid once; per-`v2`-sample evaluations reduce to `G @ h`.
    2. Batched BLAS GEMM across chunks of `v2` samples (tunable
       `chunk_v2=64` default) replaces 7 × N_v2² matvec calls.
    3. Vectorised weight-vector assembly via fancy-indexing of the
       multi-index arrays, eliminating the Python loop over basis
       terms.
    4. `numexpr` fused integrand + reduction on the hot path, falling
       back to plain NumPy when `numexpr` isn't importable.

  Combined ~30× speedup over the naive per-sample Python loop.

  Supports explicit `collimated_input=True`, `poly_order`, `n_v2`,
  `ray_field_samples` / `ray_pupil_samples`, `output_subsample`,
  `normalize_output` (`'power'` default, preserving total |E|²),
  `extract_linear_phase` (for diffractive grating surfaces at nonzero
  orders), and per-method Newton-iteration controls.

- **`apply_real_lens_traced` speedup kwargs** (opt-in, default behaviour
  unchanged):

    * `fast_analytic_phase=False` — when `True`, skips the full
      `apply_real_lens(ones, ...)` ASM-through-glass pass used to
      extract the reference phase and computes it analytically from
      per-surface sag instead (`_geometric_lens_phase` helper).
      Preserves intensity to 0.000 % and introduces <10 nm OPL
      phase error on refractive systems up to ~F/7 (Design 51 L3/L4
      scale); below the numerical noise floor for most coherent-imaging
      workflows.  Saves ~25 % wall time when `parallel_amp=False`.

    * `_Cheb2DEvaluator` refactored as array-API polymorphic with
      a combined value+gradient evaluator.  The Newton loop in
      `apply_real_lens_traced` now detects the combined-evaluation
      API (`ev_value_and_grad`) and uses it when the polynomial
      path is active, dropping from 6 evaluator calls per iteration
      down to 2 (one per coordinate) with shared Chebyshev basis
      work.  In isolated benchmarks on a 4M-sample Newton-style
      workload the refactored polynomial path runs **~12.6x faster
      than the 6-call spline baseline** (189 ms vs 2376 ms) when
      numba is installed; roughly ~4x without numba.  Combined with
      `#3` (Clenshaw-style inline Chebyshev recurrence, no
      Vandermonde materialised), `#6` (one-pass value + both
      partial derivatives), and `#1` (optional `@njit(parallel,
      fastmath)` kernel).

    * **GPU support via ``use_gpu=True``** (requires
      ``newton_fit='polynomial'`` and a working CuPy install).
      Dispatches the ``_Cheb2DEvaluator`` construction and the Newton
      inversion loop to GPU while keeping amp, amp(pw), ray-trace,
      and final field assembly on CPU (those remain CPU-only for
      now).  The polynomial fit (tiny lstsq) is always done on CPU
      to avoid a cuSOLVER dependency; only fitted coefficients +
      index arrays are pushed to the device.  The Newton loop uses
      an ``xp``-namespace throughout so the same code runs on NumPy
      or CuPy; the numba fastpath is skipped when the backend is
      CuPy and the pure-xp path (which CuPy dispatches to cuBLAS
      and elementwise kernels) runs instead.

      Validated on a 33-knot singlet fit:

          N_samples    CPU (numba)    GPU (cupy)    GPU speedup
          100,000        1.4 ms         4.8 ms       0.3x
          1,000,000     22.1 ms        15.4 ms       1.4x
          4,000,000    105.8 ms        74.1 ms       1.4x

      Modest absolute speedup at typical workloads because numba
      already parallelises the CPU path aggressively; the bigger
      payoff is for iterated workflows (many Newton calls per
      optimisation step) or very large grids (N>>4k) where CPU
      memory bandwidth starts to saturate.  Output is bit-equivalent
      to the CPU path (0.0 % RMS intensity error on Design 51 L1).

      Process-pool Newton is auto-disabled when ``use_gpu=True``
      (device arrays don't cross subprocess boundaries cheaply) and
      when ``newton_fit='polynomial'`` (the worker function currently
      only supports splines).  Both fall back to the in-process
      Newton path, which at ``ray_subsample=8`` is already fast
      enough that the pool's spawn overhead often dominates anyway.

    * `newton_fit='polynomial'` (default in 3.1.7; was `'spline'` in
      earlier versions) or `'spline'` — polynomial path replaces
      `scipy.interpolate.RectBivariateSpline` with a 2-D Chebyshev
      tensor-product fit (`_Cheb2DEvaluator`) providing the same
      `.ev(x, y, dx=0/1, dy=0/1)` API used by the Newton loop.  For
      smooth refractive lens prescriptions (all Seidel and
      higher-order aberrations are polynomials by definition),
      polynomial matches or exceeds spline accuracy on the fit and
      provides closed-form analytic derivatives.  Tunable
      `newton_poly_order=6` default (order-6 total-degree captures
      higher-order aberrations out to the 8th Seidel).  Flip back to
      `'spline'` for high-order freeforms / metasurfaces / kinoforms
      with sharp non-polynomial surface features.

    * Default `ray_subsample` bumped from `1` to `8`.  At N=32768 on
      Design 51 lenses this gives 2000–3800 samples across the
      aperture — far above the `min_coarse_samples_per_aperture=32`
      safety floor (which gave ~85 nm RMS OPD error per the library's
      internal benchmark).  At 2000 samples the projected RMS phase
      error is ~0.02 nm (λ/60 000 at 1.31 µm).  Small-grid users who
      would drop below 32 samples across the aperture are protected
      by the existing `on_undersample='error'` guardrail, which now
      raises with a message telling them to reduce to a safe value
      (typically `ray_subsample=4` or lower).

### Performance

On the Design 51 L1 benchmark (N=32768, `parallel_amp=False`, projected):

| Config | L1 time | % of baseline |
|---|---|---|
| 3.1.6 default (`ray_subsample=4`) | ~1285 s | 100 % |
| 3.1.7 default (`ray_subsample=8`) | ~1015 s | 79 % |
| + `fast_analytic_phase=True` | ~685 s | 53 % |
| + `parallel_amp=True` | ~360 s | 28 % |

Accuracy unchanged vs 3.1.6 default, within the stated tolerances.

## [3.1.6] — 2026-04-21

### Fixed

- **Zarr storage on Windows + Python 3.14 + zarr v3.**  Writing to an
  existing zarr store failed with
  ``FileExistsError: [WinError 183] Cannot create a file when that
  file already exists`` whenever ``append_plane`` or
  ``write_sim_metadata`` reopened an already-created zarr directory.
  Root cause: zarr v3's ``LocalStore._open`` unconditionally calls
  ``Path.mkdir(parents=True, exist_ok=True)`` on the store root, and
  on this specific platform combination that call raises
  ``FileExistsError`` even with the ``exist_ok`` flag -- a
  regression relative to standard Python semantics on other OSes and
  earlier Python versions.

  Fixed by adding a ``_open_zarr_group_safe`` helper that picks the
  right open-mode for the situation:

  * Store directory already exists on disk -> open with ``mode='r+'``
    (read/write, must exist), which skips the internal mkdir call
    entirely and therefore doesn't trigger the regression.
  * Store directory doesn't exist yet -> open with ``mode='a'`` to
    create it, with a ``FileExistsError`` fallback to ``'r+'`` for
    concurrent-creation races.

  The helper is used internally by ``_zarr_append_plane`` and
  ``_zarr_write_sim_metadata``; no API change for callers.  All
  existing callers (``append_plane``, ``save_planes``,
  ``write_sim_metadata``, the unified dispatch shims) benefit
  automatically.

### Compatibility

- Read-path APIs (``load_planes``, ``list_planes``,
  ``load_plane_slice``, etc.) open with ``mode='r'``, which doesn't
  mkdir and was never affected by the bug.  No changes to those code
  paths.

## [3.1.5] — 2026-04-20

### Added

- **`load_zmx_prescription` and `load_zemax_prescription_txt` now
  return `prescription['object_distance']`.**  Zemax sequential files
  typically carry non-refractive "dummy" surfaces between the object
  (or STOP / source plane) and the first active lens surface —
  coordinate breaks, field-reference planes, MLA mounting planes,
  etc.  Previous loader versions filtered these out and discarded
  their `DISZ` (z-thickness) values, which meant any wave-optics
  simulation driven by the returned prescription implicitly placed
  its source field AT the first refractive surface, collapsing that
  design-intended obj-space geometry.  The symptom was a
  defocus-like blur at the downstream image plane proportional to
  the dropped distance.

  The new key `object_distance` (float, meters) is the sum of `DISZ`
  values from the STOP surface up to but not including the first
  active surface.  If the file has no STOP, the sum runs from SURF 0
  onward.  Non-finite `DISZ` (`INFINITY`) contributes 0 so
  object-at-infinity configurations behave the same as before.

  Downstream callers driving a simulation from the loaded
  prescription should now propagate their source field by
  `prescription['object_distance']` before invoking the first lens
  operator to recover the .zmx's original paraxial geometry.

  **Detection example (Design 51 tx4designstudy51.zmx):**

      rx = la.load_zmx_prescription('tx4designstudy51.zmx')
      print(rx['object_distance'])   # -> 0.096669 (m), = 96.67 mm

  of previously-dropped dummy-surface thickness, which (without
  this change) caused ~235 µm of defocus blur on each MLA-
  collimated beam at the metasurface plane.

### Compatibility

- The new key is additive; all existing callers continue to work
  unchanged.  Prescriptions built manually (`make_singlet`,
  `make_doublet`, `make_cylindrical`, etc.) don't set
  `object_distance` at all — callers reading this key should use
  `rx.get('object_distance', 0.0)` for safety.

## [3.1.4] — 2026-04-18

### Changed (default flip)

- **`apply_real_lens_traced(..., tilt_aware_rays=...)` default changed
  from ``True`` to ``False``.**  The Tier 1 input-aware ray launch
  added in 3.1.2 was meant to extract per-pixel ray directions from
  the input field's local phase gradient so the lens OPL would reflect
  actual angles of incidence.  In combination with the (also default)
  ``preserve_input_phase=True`` path, however, it creates a
  reference-frame inconsistency that does not affect single-mode
  plane-wave-like inputs but produces materially wrong output on
  multi-mode inputs (post-DOE diffraction patterns, compound
  superpositions).

  Specifically, the ``preserve_input_phase`` output is assembled as

        E_out = E_analytic * exp(i * delta_phase)
        delta_phase = k0 * opl_traced - phase_analytic_lens

  where ``phase_analytic_lens`` is extracted by running
  ``apply_real_lens`` on a unit plane wave -- a plane-wave reference.
  For ``delta_phase`` to be a clean "ray-traced minus analytic"
  correction, ``opl_traced`` must share that reference (rays launched
  collimated at the entrance).  With ``tilt_aware_rays=True``,
  ``opl_traced`` is instead evaluated at per-pixel launch angles; for
  multi-mode inputs those angles vary wildly across the pupil,
  ``delta_phase`` mixes lens-model correction with tilt-induced phase
  shifts the plane-wave reference does not contain, and the output
  field collapses to a "power-lost-to-bandlimit" state (TX Design 36
  rerun on 2026-04-17 showed 0.55 % power conservation on a 4-lens
  post-DOE system, vs 92.5 % with the plane-wave default).

  The 3.1.3 multi-mode Gaussian-smoothing of the extracted tilts was a
  mitigation for the pathological gradient aliasing but could not
  address the underlying reference-frame inconsistency -- with enough
  smoothing the tilts collapse toward zero anyway, and the legitimate
  per-order tilt structure of the post-DOE field is destroyed in the
  process.  Flipping the default to ``False`` side-steps the whole
  issue by using the reference-consistent plane-wave launch that
  pre-3.1.2 releases used.  Users with specifically small, uniform
  input tilts who want the per-ray OPL variation can still pass
  ``tilt_aware_rays=True`` explicitly and validate on their case.

  The 3.1.3 ``_sample_local_tilts`` Gaussian smoothing + ``max_sin``
  clip stay in the library -- they are still consulted when
  ``tilt_aware_rays=True`` is explicit or when the experimental
  ``inversion_method='backward_trace'`` needs an exit-direction
  estimate.  They just don't run on the default path anymore.

### Performance

- **Paraxial-magnification Newton initial guess in
  `apply_real_lens_traced`.**  The pre-3.1.4 initial guess was a
  hard-coded ``(xe, ye) = 1.10 * (Xw, Yw)`` (implicitly assuming
  every lens has a paraxial magnification of ~0.91 at its exit
  vertex).  For singlets this was approximately right; for compound
  systems with real imaging magnification (TX Design 36 full-system
  inversion has M = 0.25) it puts Newton 4x from the answer and costs
  several extra iterations per pixel.

  The new path measures the per-lens paraxial magnification directly
  from the already-computed forward-map slope at the central launch
  point (the central-finite-difference ``dx_out/dx_in`` and
  ``dy_out/dy_in``) and uses ``(Xw/M_x, Yw/M_y)`` as the initial
  guess.  Zero additional compute (the values are already in the
  forward-trace output array), and for singlets the result is
  essentially the same as before; for compound-system callers the
  speedup is several iterations.  Parallel-pool workers get the
  inverse-magnification factors through ``_spline_data`` so the
  in-process serial path and the out-of-process chunked path seed
  Newton identically.

- **`inversion_method='backward_trace'` opt-in on
  `apply_real_lens_traced`** (experimental).  Replaces the forward
  ray trace + Newton spline inversion with a single backward pass
  from a coarse subsample of the exit-plane wave grid, driven by
  the phase-gradient-extracted exit direction.  Validated to
  reproduce the Newton path's OPL to sub-picometre on a plano-convex
  singlet single-ray test; end-to-end ``apply_real_lens_traced`` at
  N=1024 shows ~30 nm OPD RMS agreement and ~3.2x speedup vs the
  Newton default.  Accuracy is bounded by the finite-difference +
  smoothed phase-gradient direction estimate, not by the reversal
  itself.  Default stays ``'newton'`` while the backward path is
  validated on a wider set of prescriptions; opt in by passing
  ``inversion_method='backward_trace'`` to trade some accuracy
  (~30 nm OPD per lens) for a substantial speedup on large grids.

## [3.1.3] — 2026-04-17

This release is a set of targeted performance, robustness, and
precision-flexibility improvements to the hot path for N=32768
coherent propagation runs.  All changes are backwards-compatible
defaults (bit-identical at complex128 with no new kwargs passed)
except where explicitly noted.

### Performance

- **Numexpr-fused phase-screen multiply in `apply_real_lens`.**  The
  per-surface `E * np.exp(-1j * k0 * opd)` step used to materialise
  three complex128 NxN intermediates (the broadcast `-1j*k0*opd`,
  the `exp()` output, and the multiply result -- ~50 GB of churn
  per surface at N=32768).  When `numexpr` is available (optional
  dependency `[perf]`) and the field has at least 2^20 elements, the
  library routes the expression through `ne.evaluate('E * exp(-1j*k0*opd)', out=E)`
  instead.  Threaded, chunked, fully in-place, and numerically
  identical to the numpy path at double precision (`max |diff| = 0`
  on a singlet test).  Measured 1.66x on `apply_real_lens` at N=4096.
  Automatic fallback to the numpy path when numexpr is not installed
  or when the field is too small for the overhead to pay off.
  New top-level export: `NUMEXPR_AVAILABLE`.

- **Decenter-aliased entrance grids in `apply_real_lens`.**  When a
  surface has `decenter == (0, 0)` (the common case), the library
  now aliases `Xs = X`, `Ys = Y`, `h_sq = h_sq_axis` instead of
  allocating three new float64 NxN arrays per surface (~24 GB at
  N=32768).  Safe because downstream code only reads these arrays
  and creates fresh arrays for `sag + tilt[i]*Xs` / `sag + form_err`.

- **Single-slot pyFFTW plan cache with per-plan threading.Lock in
  `_fft2` / `_ifft2`.**  Replaces the previous
  `pyfftw.interfaces.numpy_fft` shim, which allocated fresh 16 GB
  aligned buffers on every call and held them for 30 s -- the root
  cause of the Windows contiguous-address-space fragmentation that
  forced `USE_PYFFTW = False` as a workaround on N=32768 runs.  The
  new path holds exactly one `pyfftw.FFTW` plan per direction (forward,
  inverse), each backed by an in-place 16 GB aligned buffer that is
  allocated once at first use and reused for the lifetime of the
  process.  Shape/dtype/threads are keyed on the cache; any mismatch
  drops the old plan + buffer (GC'd) and reallocates.  A per-plan
  `threading.Lock` serialises concurrent execution from parallel
  callers (e.g. the `parallel_amp` path in `apply_real_lens_traced`).
  Measured 1.50x on an N=4096 ASM call.  `reset_fft_backend()` now
  clears the new plan slots in addition to the bad-shape blacklist.

- **Parallelised `amp` and `amp(pw)` passes in `apply_real_lens_traced`**
  via a 2-worker `ThreadPoolExecutor`.  The two internal
  `apply_real_lens` calls are data-independent and run concurrently;
  FFT execution is serialised through the per-plan lock above but
  non-FFT work (sag, phase screens, glass-interval setup) overlaps.
  Measured 1.56x on the combined amp step when isolated at N=4096.
  Opt-out via new kwarg `parallel_amp=False`.  Memory guard via
  `parallel_amp_min_free_gb=48.0` auto-disables when RAM is tight
  (doubled per-call transient working set: ~2x the peak of a single
  `apply_real_lens` at the same grid size).

- **Amplitude-masked Newton inversion in `apply_real_lens_traced`.**
  New kwargs `newton_amp_mask_rel=1e-4`, `newton_mask_dilate_coarse_px=2`.
  The Newton-inverted entrance→exit spline is now evaluated only on
  coarse-grid pixels where the analytic amplitude envelope exceeds
  `newton_amp_mask_rel * amp.max()`, with the mask dilated by
  `newton_mask_dilate_coarse_px` coarse pixels so bilinear
  interpolation near mask boundaries always has real data in its
  support.  Skipped pixels get NaN and are handled by the existing
  NaN-propagation path -- identical to the ray-domain-failure
  handling already in place.  Self-disables when the mask would
  capture >95 % (overhead not worth it) or <1 % (pathological; fall
  back to full-grid Newton).  Biggest benefit on post-DOE fields
  where only the diffraction-order pixels are bright.

### Added

- **Complex-dtype awareness throughout the critical path.**  The
  `apply_real_lens`, `apply_real_lens_traced`, `apply_mirror`, and
  `angular_spectrum_propagate` functions now preserve the caller's
  complex dtype (complex64 or complex128) rather than forcing
  complex128 internally.  Module-level default
  `DEFAULT_COMPLEX_DTYPE = np.complex128` controls the fallback
  when a non-complex input is given (used by a handful of builders
  elsewhere in the library).  Runners can opt into complex64 mode
  for ~2x memory and throughput by creating their fields as
  complex64 from the start; all library functions on the hot path
  then stay at that precision end-to-end.

  To keep complex64 accurate despite the huge phase magnitudes
  (`k * z` reaches ~4e5 rad over an 80 mm air gap at
  `lambda = 1.31 um`), **kernel-phase and phase-screen arguments are
  always computed in float64 and reduced modulo 2 pi before the
  final trig cast to float32**.  Without this mitigation the naive
  complex64 ASM kernel would inject ~0.02 rad noise per Fourier
  pixel (a diffuse speckle floor at ~-80 dB); with it, the only
  remaining precision loss is the FFT's natural single-precision
  round-off.  Validated on N=512:

  | Test | Kernel phase range | c128 vs c64 rel err |
  |---|---|---|
  | ASM z=1 mm | ~5 rad | 2.2e-7 |
  | ASM z=80 mm | ~4e5 rad | 3.1e-7 (mitigation working) |
  | `apply_real_lens` singlet | ~240 rad / surface | 2.4e-7 |
  | End-to-end 2-lens chain | mixed | 4.9e-7 |

  New top-level export: `DEFAULT_COMPLEX_DTYPE`.

- **`smooth_sigma_px` kwarg on `_sample_local_tilts` (default 4.0)**
  for robustness on multi-mode inputs.  The Tier 1 input-aware ray
  launch added in 3.1.2 extracts per-pixel tilts from the local
  phase gradient; on a single-mode field (plane wave, Gaussian,
  MLA-tilted beamlet) this produces a smooth tilt field that
  correctly parametrises the forward ray trace.  On a multi-mode
  interferogram (post-DOE field with 144 diffraction orders) the
  phase gradient aliases at every fringe boundary, clips to
  `max_sin`, and injects chaotic per-pixel directions into the
  ray trace -- the `RectBivariateSpline` over the resulting
  entrance→exit map becomes high-frequency, Newton diverges, and
  the output field collapses to zero.  The new amplitude-weighted
  Gaussian smoothing ( `blur(|E|^2 * L) / blur(|E|^2)` ) low-passes
  the tilt field before clipping:

  *   Single-mode fields: tilt magnitudes preserved to ~1 % (σ=4 px
      is much smaller than the beam feature scale).
  *   Multi-mode superposition: oscillations average to mean tilt,
      which for a balanced DOE is near zero -- naturally degenerating
      to the classical collimated launch.
  *   Mixed fields: handled per-pixel via the local-neighbourhood
      average.

  Pass `smooth_sigma_px=0` to recover the pre-3.1.3 behaviour.
  Optional `multimode_diagnostic` dict parameter, populated with
  `raw_rms_L`, `smoothed_rms_L`, `smoothing_ratio` etc., for callers
  that want to log or assert on the smoothing's effect.

### Fixed

- **`apply_real_lens_traced` no longer produces a zero output field
  on multi-mode inputs at large N.**  Symptom: at N=32768 with a
  post-DOE input (12×12 Dammann orders), the Newton inversion
  diverged for all pixels -- chunk 1 took 100+ minutes iterating,
  subsequent chunks finished in seconds as every ray hit the
  fit-domain clip and was NaN'd out.  Root cause: the aliased
  per-pixel tilts from `_sample_local_tilts` fed chaotic
  `(x_out, y_out)` entries to the entrance→exit spline; the
  cubic-spline fit overfit to the high-frequency oscillations and
  its derivatives (used in Newton's Jacobian) became unusable.  Fix:
  amplitude-weighted Gaussian smoothing of the tilt field described
  above.  Validated on the plane_09 field from the TX Design 36
  production run: max |L| dropped from 0.5 (clipping active) to 0.22
  (no clipping), spline derivatives stay sensible, Newton converges
  normally.

- **Analysis pool zombie cleanup in `tx_design_study_analysis.run_all_analysis`**
  (note: this fix is in the Reverse_Symmetric_ASM tree, not the
  library itself, but is relevant to users of the analysis pattern).
  Workers previously stayed resident after the `with
  ProcessPoolExecutor` block closed -- matplotlib figure state and
  MKL thread-pool atexit handlers blocked `_process_worker`'s
  `sys.exit()` on Windows spawn, leaving multi-GB zombies pinning
  RAM.  Now uses `max_tasks_per_child=1` (forces worker OS exit
  after one task), `plt.close('all')` + `gc.collect()` in each
  worker's `finally`, and a 30 s bounded shutdown wait with
  SIGTERM/SIGKILL straggler sweep if the primary shutdown hangs.

### Added (exposed)

- `DEFAULT_COMPLEX_DTYPE` -- module-level precision default, exported
  from the top-level package.
- `NUMEXPR_AVAILABLE` -- optional-backend availability flag,
  exported from the top-level package (useful in runner scripts
  that want to gate behaviour on the fast path being available).

## [3.1.2] — 2026-04-17

### Added

- **pyFFTW-to-scipy.fft automatic fallback on allocation failure.**
  At very large grids (e.g. ``N = 32768``) and tight RAM, pyFFTW's
  per-shape aligned-buffer plan can fail to allocate (the Windows
  allocator can't find a contiguous ~16 GB block even when total
  free RAM looks sufficient).  ``_fft2`` / ``_ifft2`` now wrap the
  pyFFTW call in try/except, catch any exception, emit a one-time
  ``RuntimeWarning``, blacklist that shape for the remainder of the
  session, flush the pyFFTW plan cache, and fall through to
  ``scipy.fft`` (numerically identical to pyFFTW at 1e-14 noise,
  just ~6x slower on large grids without aligned buffers).

  Three new user-facing controls:

  * ``op.set_fft_fallback(False)`` -- disable the fallback and let
    pyFFTW errors propagate (useful to flush out genuine backend
    bugs).
  * ``op.reset_fft_backend()`` -- clear the bad-shape blacklist and
    the pyFFTW plan cache, so subsequent calls retry pyFFTW (use
    after a big one-off allocation has been freed).
  * Top-level exports: ``set_fft_fallback``, ``reset_fft_backend``.

- **Tier 1 input-aware ray launch** (new ``tilt_aware_rays`` kwarg,
  default True) in :func:`apply_real_lens_traced`.  Each ray's
  initial direction cosines ``(L, M)`` are now derived from the
  local phase gradient of ``E_in`` at its entrance position so the
  lens OPL is evaluated at the ACTUAL angle of incidence rather
  than under a blanket plane-wave assumption.  For plane-wave
  inputs this is bit-identical to the collimated launch; for MLA /
  DOE / off-axis / pre-aberrated inputs it correctly carries the
  lens-OPL-vs-angle dependence through to the exit plane.  Tilts
  are clipped to ``|sin(theta)| <= 0.5`` for numerical safety and
  low-amplitude pixels (< 0.1 % of peak) are treated as noise
  floor and launched collimated.  Cost overhead: a few-percent
  (one numpy gradient + bilinear resample), bit-exact equivalence
  on plane-wave inputs verified.

### Fixed

- **`apply_real_lens_traced` was silently discarding the input
  field's phase.**  Before this fix the output was
  ``|apply_real_lens(E_in)| * exp(i*k0*OPL_traced)`` -- the
  amplitude carried the input correctly but the phase only reflected
  the lens's ray-traced OPL applied to a synthetic plane wave.  Any
  input-field phase structure (source tilt, MLA / DOE modulation,
  off-axis wavefronts, pre-applied aberrations) was dropped.
  Symptom: tilted inputs focused on-axis; MLA-modulated inputs came
  out as featureless envelopes at downstream planes.

  New default ``preserve_input_phase=True`` keeps the full complex
  ``E_analytic`` and applies a correction that replaces the analytic
  model's lens phase with the ray-traced OPL.  Matches
  :func:`apply_real_lens`'s behaviour for the input-field part, with
  the ray-traced OPL correction on top.

  Cost: runs ``apply_real_lens`` a second time on a unit plane-wave
  reference so we can extract and subtract the analytic lens phase.
  ~20 % overhead on the total function time at large N.

  Pass ``preserve_input_phase=False`` to restore the legacy behaviour
  (useful for plane-wave-only lens-OPD measurements where the
  distinction is moot).

### UI-library fix

- **Asphere coefficient editor** (``ui/surface_editors.py``) was
  storing ``aspheric_coeffs`` as a Python ``list`` but the library's
  canonical convention (used by Zemax import,
  :func:`surface_sag_general`, and the raytracer) is a ``dict``
  ``{power: coeff}``.  UI-edited aspherics would crash with
  ``AttributeError: 'list' object has no attribute 'items'`` when
  the prescription was then simulated.  Now stores the correct dict
  form and migrates legacy list-format on load.

## [3.1.1] — 2026-04-16

### Performance

- **`apply_real_lens_traced` Newton inversion now parallel.**  The
  `n_workers` kwarg (previously a dead parameter) now dispatches the
  embarrassingly-parallel Newton-invert step to a
  `ProcessPoolExecutor`.  Each worker rebuilds the three
  `RectBivariateSpline` objects from their knot data locally, so the
  pickling cost per chunk is just the knot arrays (~200x200 floats),
  not the spline objects themselves.
  - Measured: 8.3x speedup on 16 workers on a 4M-pixel Newton
    benchmark.
  - On small grids (< 200 k coarse pixels) the function auto-falls
    back to the in-process serial path so pool startup doesn't make
    small calls slower.
  - Numerically identical to the serial path (verified `max |diff|
    = 0` on a 1 Mpx test).
  - Threading is explicitly **not** used: SciPy's
    `RectBivariateSpline.ev` does not release the GIL in current
    SciPy versions, contrary to the previous docstring claim.  The
    previous comment about thread scaling has been removed.

### Added

- **`min_coarse_samples_per_aperture` + `on_undersample` kwargs on
  `apply_real_lens_traced`.**  Subsampling guardrail: if the
  coarse Newton grid has fewer than N samples across the lens
  aperture, the cubic-spline interpolation of the wavefront aliases
  and the result is wrong.  Benchmarks showed the rule is roughly
  `RMS phase err ~ (coarse_samples_per_aperture)^-2`:

  | Coarse samples / aperture | Typical RMS phase err (lambda=1.31 um) |
  |---|---|
  | 64 | ~20 nm |
  | 32 (default threshold) | ~85 nm |
  | 16 | ~350 nm (unusable) |

  Default policy is `on_undersample='error'`: raises `ValueError`
  with the safe `ray_subsample` value computed for the current
  grid, so the user is never silently running a corrupt sim.
  `'warn'` and `'silent'` policies are available; setting
  `min_coarse_samples_per_aperture=0` disables the check.

## [3.1.0] — 2026-04-16

UX-pass companion release: progress hooks, `'real_lens_traced'`
element type for `propagate_through_system`, codegen promoted to the
public API, and a handful of UI-driven correctness fixes.

### Added (new merit term + multi-prescription optimisation)

- **`optimize.MatchIdealSystemMerit`** -- propagate a reference source
  through BOTH an idealised thin-lens architecture and the real
  prescription, then penalise the output-field mismatch.  Unlike
  `MatchIdealThinLensMerit` (which compares exit-pupil OPD to a bare
  converging sphere), this merit compares the actual complex output
  fields, so the optimizer drives the real lens toward matching the
  ideal's radiation pattern AND relative phase -- which is what the
  "replace this thin-lens system with a real one" workflow actually
  wants.  Supports four similarity metrics
  (`field_overlap`, `field_mse`, `intensity_mse`, `intensity_overlap`),
  arbitrary architectures via element lists, optional pre/post
  propagations, and a `single_lens(f)` convenience factory for the
  common case.  Also supports:
    * ``focus_search=True`` -- axial z-offset scan that decouples
      "correct focal plane" from "aberration quality".  Measured
      improvement: 93 % penalty reduction on a thick plano-convex
      whose BFL sits 6.7 mm off the ideal's target, letting the
      optimizer converge on shape rather than fighting the BFL shift.
    * ``wavelengths=[...]`` -- built-in chromatic sweep; evaluates
      at each wavelength and averages the penalty.
    * ``field_angles=[(theta_x, theta_y), ...]`` -- built-in off-axis
      sweep; applies a linear-phase carrier to the source for each
      field point.  Combines Cartesian-product-wise with
      ``wavelengths``.
- **`optimize.MultiPrescriptionParameterization`** -- holds a list of
  template prescriptions and a free-var list whose entries start with
  a prescription index.  ``design_optimize`` recognises the class
  automatically, populates ``ctx.prescriptions`` alongside the
  existing single ``ctx.prescription`` (which stays == ``[0]`` for
  backward compatibility), and passes both through to the merit
  terms.  ``MatchIdealSystemMerit``'s ``_prescription_`` placeholder
  now accepts an ``'index'`` key to select which prescription slots
  in where.  Verified with a 4f architecture: two singlets jointly
  optimised, 220 iterations, merit reduces from 0.060 -> 0.005, each
  lens settles at a distinct optimised form.

### Fixed (third-pass deep-audit findings)

- **`analysis.compute_psf`** -- default normalisation changed from
  ``'peak'`` (which made ``psf.max() == 1`` unconditionally,
  silently breaking every canonical
  ``strehl = compute_psf(abb).max() / compute_psf(ideal).max()``
  pattern) to ``'power'``.  A direct test now recovers
  Strehl = 0.906 / 0.674 / 0.411 / 0.206 at 0.05 / 0.10 / 0.15 /
  0.20 waves RMS of Z(4,0), tracking extended Marechal to ~1 %
  at small aberrations.  The old behaviour is available as
  ``normalize='peak'`` (for display) and raw ``|FFT|^2`` as
  ``normalize='none'``.  Breaking-change advisory: callers who
  relied on peak-normalised output must now opt in explicitly.
- **`propagation.angular_spectrum_propagate_tilted`** -- band-limit
  mask now tests ``|FX| < fx_max`` on the baseband (post-demod)
  frequency grid instead of ``|FX_shifted|``.  For any non-trivial
  tilt the old mask was zeroing the baseband DC mode and killing
  the propagated field.  Measured power preservation after the fix:
  input ``mean|E|^2 = 0.235``, after ``z = 0.01 m`` with tilt 0.05
  rad and bandlimit=True: ``mean|E|^2 = 0.235`` (was 3.7 x 10^-7).
- **`optimize.FocalLengthMerit` / `BackFocalLengthMerit`** -- guard
  against ``target == 0`` (collimator / afocal case).  Old code
  computed ``(efl - 0) / 0`` -> NaN and poisoned the optimizer.
  New behaviour: when ``target == 0``, penalise ``efl^2`` directly,
  driving the optimizer toward infinite EFL.
- **`optimize.ToleranceAwareMerit`** -- each Monte-Carlo trial now
  recomputes the perturbed system's BFL via ``system_abcd`` and
  scans through focus around THAT plane rather than the nominal
  BFL.  Perturbations that significantly shift the focal plane
  (e.g. large first-surface decenters) no longer produce
  artificially low Strehl from scanning off-focus.
- **`optimize.ToleranceAwareMerit`** -- form-error seed is now
  deterministic per ``(trial, surface_index)`` pair instead of
  drawn from the global RNG.  Two runs with the same base seed
  now produce identical perturbation realisations regardless of
  surrounding RNG calls.
- **`optimize`: new ``ctx_is_valid`` helper + merit-term sentinel
  guards** -- when ``system_abcd`` fails, EvaluationContext sets
  ``efl = bfl = 1e9``; merit terms that consume those now return a
  bounded penalty instead of ``(1e9 - target)^2`` which dragged the
  optimizer away from good regions.
- **`doe.makedammann2d`** -- target-pattern centering uses plain
  integer division ``(N_big - N_small) // 2`` instead of the
  Octave-port's ``ceil((N_big - N_small)/2) + 1`` offset.  For odd
  differences the old code placed the input pattern one cell to
  the left of center, breaking the binary symmetry Dammann designs
  rely on.
- **`storage.set_storage_backend('zarr')`** -- raises ``ImportError``
  immediately if zarr isn't installed instead of lazy-failing on
  the first ``append_plane`` call.
- **`codegen`** -- generated-script numeric literals now use
  ``.17e`` format (full IEEE 754 round-trip precision) instead of
  ``.6e`` which lost ~0.1 nm per value; ~0.1 um drift across a
  multi-surface prescription.
- **`elements.apply_aperture` / `apply_gaussian_aperture`** --
  accept an optional ``dy`` kwarg (defaults to ``dx``) for
  rectangular (non-square) grids.  Previously the y-coordinate
  grid was built from ``dx`` regardless, silently stretching the
  aperture along the y axis on non-square grids.
- **`optimize.RMSWavefrontMerit`** -- exposed the previously
  hard-coded ``exclude_low_order`` via a constructor kwarg.
  Default value raised from 3 to 4 (piston + 2 tilts + defocus)
  to match the "image-quality RMS after best-focus" convention.
- **`through_focus.plot_through_focus`** -- Strehl axis uses
  ``max(1.05, 1.1 * Strehl_max)`` so rare super-unity peaks stay
  visible.
- **`user_library`** -- serialise/deserialise pair now recursively
  handles ``float('inf')`` / ``float('-inf')`` anywhere in the
  prescription (previously only ``surfaces[i]['radius']`` was
  restored; infinities in ``thickness``, ``aperture_diameter``,
  ``conic``, etc. came back as the string ``'Infinity'`` and
  caused downstream ``TypeError``).
- **`elements.apply_zernike_aberration`** docstring -- explicit
  unit-conversion example for round-tripping with
  ``analysis.zernike_decompose`` (apply takes waves, decompose
  returns metres).
- **`hdf5_io`** no longer re-exports the private
  ``storage._decode_attr``.

### Fixed (second-pass deep-audit findings)

- **`rcwa.py`** — the old `rcwa_1d` built the Moharam-Gaylord
  Fourier eigendecomposition but then **threw it away** and
  renormalised ``T / sum(T)``, hiding any non-unit energy in R.
  Replaced with a clean analytical thin-phase-grating formula that
  respects energy conservation (`sum|t_m|^2 = 1` by Parseval for
  propagating orders; evanescent orders correctly carry zero).
  Docstring now states up front that this is a scalar thin-grating
  approximation, not full RCWA -- R is always zero in the thin
  regime, and a future S-matrix implementation is left as a clear
  TODO.  The function signature is unchanged; existing call sites
  keep working.
- **`detector.py::apply_detector`** — pixel binning used
  integer-truncation indexing that gave wildly non-uniform
  per-pixel sample counts (9 / 12 / 16 / 20 / 25 depending on where
  integer boundaries fell).  On a uniform field the resulting
  "std / sqrt(mean)" was ~20 instead of the ~1 a pure Poisson
  process should give, so shot-noise statistics were unreliable.
  Replaced with `scipy.ndimage.zoom`-based area-weighted
  integration.  Measured std / sqrt(mean) on a uniform high-count
  field is now 1.03 (integer pitch ratio) and 1.04 (non-integer).
- **`freeform.py::surface_sag_chebyshev`** — outside the Chebyshev
  normalisation box `[-norm_x, norm_x] * [-norm_y, norm_y]`, the
  function used to return the boundary value `T_n(+-1)`, creating
  a large step discontinuity at the domain edge that broke the ray
  tracer's Newton intersection solver.  Now zeroes the departure
  outside the domain while preserving the base conic sag.
- **`interferometry.py::phase_shift_extract`** — added
  ``convention='hardware' | 'library'`` kwarg.  The extraction
  formula's sign depends on whether the caller supplies frames
  following the Schwider/Hariharan convention ``I = a + b*cos(phi - s)``
  (what every real phase-shifting interferometer produces) or the
  opposite ``I = a + b*cos(phi + s)`` (what this library's own
  `simulate_interferogram` produces).  Default is ``'hardware'`` so
  that real-instrument data round-trips sign-correctly; pass
  ``'library'`` to round-trip a `simulate_interferogram` output.
- **`multiconfig.py::keplerian_telescope` and
  `beam_expander_prescription`** — the thin-lens separation
  ``f_obj + f_eye`` was used even though the functions build thick
  singlets, leaving the output systems non-afocal (|C| ~ 0.07/mm
  on a Keplerian 200/25 mm).  Replaced with a one-step linear
  solve on the air gap that drives the system ABCD's C element
  to exactly zero (machine precision).  Separately fixed a geometry
  bug in `beam_expander_prescription` where the eyepiece was built
  as plano-concave `[R, inf]` while using the equi-convex focal-
  length formula `R = f*(n-1)*2`, which halved the eyepiece focal
  length and gave an expansion ratio of M/2.  Eyepiece is now
  equi-shaped ``[R, -R]`` matching the formula.  `M=5` now
  delivers 5x (was 2.5x).
- **`multiconfig.py::afocal_angular_magnification`** — the
  ``is_afocal`` test used ``|B| < 1e-6``, but the afocal condition
  is ``C = 0`` (collimated in -> collimated out), not ``B = 0``
  (which is 1:1 imaging).  Test now uses ``|C| * aperture_radius <
  1e-6`` (equivalent to a sub-microradian residual output
  divergence for a typical input bundle).

### Added

#### Progress hooks (`progress.py`, new module)
- `ProgressCallback` type alias, `call_progress(cb, stage, frac, msg)`
  helper, and `ProgressScaler` for nesting sub-tasks within a parent
  budget.
- The following long-running core functions now accept an optional
  `progress=cb` keyword:
  - `apply_real_lens` (per-surface progress)
  - `apply_real_lens_traced` (amp pass + ray trace + Newton inversion)
  - `propagate_through_system` (per-element, recursively scaled into
    the lens-model sub-progress where applicable)
  - `through_focus_scan` (per-z-plane)
  - `tolerancing_sweep` (per-perturbation, each sub-run scaled to
    its slice of the overall bar)
  - `monte_carlo_tolerancing` (per-trial)
  - `design_optimize` (per merit-function evaluation; approximate
    because scipy's optimizers don't expose uniform iteration counters)
- Hooks are completely opt-in (None = no overhead) and exception-safe
  (a broken callback can never crash a simulation).  Signature is
  `(stage: str, fraction: float, message: str = '')`.
- Shared between any script that wants a progress bar and the
  optical-designer UI's wave-optics / optimizer / tolerance docks.

#### System propagation
- `propagate_through_system` gained a `'real_lens_traced'` element
  type that delegates to `apply_real_lens_traced` with optional
  `bandlimit` and `ray_subsample` per-element overrides.
- Pass-through `progress=` to `apply_real_lens` /
  `apply_real_lens_traced` per element via `ProgressScaler` windows.

#### Code generation (`codegen.py`)
- `generate_simulation_script`, `generate_script_from_zmx`, and
  `generate_script_from_txt` are now exported from the top-level
  `lumenairy` namespace.  Previously importable but
  undocumented.

### Changed

- `make_singlet` and `make_doublet` now always emit `radius_y`,
  `conic_y`, and `aspheric_coeffs_y` keys (set to `None`) so a
  prescription dict round-trips through diff-friendly tooling
  without ambiguity.
- `propagation.py` module docstring spells out the return-type
  contract (ASM + RS bare arrays; Fresnel + Fraunhofer 3-tuples).
- `apply_real_lens_traced` docstring surfaces the
  `dx \u2264 \u03bb*f / aperture` Nyquist requirement and points readers at
  `check_opd_sampling`.
- `elements.zernike` docstring corrected: it uses OSA / unit-variance
  normalization with `(n, m)` indexing, not Noll single-index.
- `create_multi_field_sources` docstring flags its
  list-of-tilted-plane-waves return shape (different from the scalar
  `create_*` helpers' `(E, x, y)` triple).

### Fixed

- `analysis.remove_wavefront_modes` accepts a `weights` keyword for
  intensity-weighted least-squares fits; vignetted / annular pupils
  no longer leak high-order content into piston/tilt/defocus.
- `raytrace.surfaces_from_prescription` plumbs the optional
  `freeform` key through to the `Surface` dataclass; freeform sags
  (XY-polynomial / Zernike / Chebyshev) are now ray-traceable, not
  wave-only.

### Removed

- `optical_table.py` + `optical_table.html` and their bidirectional
  scene/element/prescription translators.  The bundled HTML simulator
  was unreferenced from the GUI and the only Python entry points were
  not exported from `__init__.py`, so it was effectively dead code.
  Daniel L. Marks' zlib-licensed HTML application is no longer
  redistributed; see prior commits for the source.

---

## [3.0.0] — 2026-04-16

Major release: hybrid wave/ray lens model, 15+ new physics modules,
design optimizer, comprehensive validation suite.

### Added

#### Lens modelling
- `apply_real_lens_traced` — hybrid wave/ray lens model combining
  wave-optics amplitude (from `apply_real_lens`) with geometrically
  exact per-pixel ray-traced OPL phase.  Sub-nanometre OPD agreement
  with the geometric ray trace across all tested lens geometries
  (singlets, doublets, meniscus, biconcave, equi-convex).
- `surface_sag_biconic` — biconic/cylindrical/toroidal surface sag
  with independent x/y radii, conics, and aspheric coefficients.
- `apply_cylindrical_lens` — cylindrical thin-lens phase screen.
- `apply_grin_lens` — gradient-index rod lens.
- `apply_axicon` — conical phase element.
- Prescription keys `radius_y`, `conic_y`, `aspheric_coeffs_y` for
  anamorphic surfaces throughout the pipeline (ray tracer, ABCD,
  Seidel, OPD analysis).

#### Prescription builders
- `make_cylindrical` — cylindrical singlet prescription.
- `make_biconic` — biconic singlet prescription.
- `export_zemax_lens_data` — human-readable Zemax LDE text export.
- `export_zemax_zmx` — Zemax .zmx binary prescription export.

#### Design optimizer (`optimize.py`, new module)
- `DesignParameterization` — maps a flat parameter vector to a
  prescription dict for scipy optimizers.
- 18 merit term classes: `FocalLengthMerit`, `BackFocalLengthMerit`,
  `SphericalSeidelMerit`, `StrehlMerit`, `RMSWavefrontMerit`,
  `SpotSizeMerit`, `ChromaticFocalShiftMerit`, `MatchIdealThinLensMerit`,
  `MatchTargetOPDMerit`, `ZernikeCoefficientMerit`, `CompositeMerit`,
  `CallableMerit`, `MultiWavelengthMerit`, `MultiFieldMerit`,
  `MinThicknessMerit`, `MaxThicknessMerit`, `MinBackFocalLengthMerit`,
  `MaxFNumberMerit`, `ToleranceAwareMerit`.
- `design_optimize` — unified entry point supporting L-BFGS-B, SLSQP,
  trust-constr, differential_evolution, basin_hopping, dual_annealing,
  and Levenberg-Marquardt (via Householder QR).

#### Through-focus and tolerancing (`through_focus.py`, new module)
- `through_focus_scan` — propagate a field to multiple z-planes and
  collect peak intensity, Strehl, and beam metrics at each.
- `find_best_focus` — locate best focus from a through-focus scan.
- `plot_through_focus` — matplotlib visualization.
- `diffraction_limited_peak` — ASM-based reference for Strehl.
- `Perturbation`, `apply_perturbations` — structured perturbation model.
- `tolerancing_sweep` — systematic single-parameter sensitivity.
- `monte_carlo_tolerancing` — Monte Carlo tolerance analysis.

#### Analysis (`analysis.py`, expanded)
- `zernike_decompose` — Householder QR with column pivoting (gelsy),
  numerically stable for high-order and partial-pupil data.
- `zernike_reconstruct`, `zernike_polynomial`, `zernike_basis_matrix`.
- `zernike_index_to_nm`, `zernike_nm_to_index` — OSA index helpers.
- `chromatic_focal_shift` — focal shift vs wavelength.
- `polychromatic_strehl` — polychromatic Strehl ratio.
- `check_opd_sampling` — Nyquist margin calculator for OPD extraction.
- `wave_opd_1d`, `wave_opd_2d` — with focal-length warnings and
  optional reference-sphere subtraction (`f_ref`).

#### Vector diffraction (`vector_diffraction.py`, new module)
- `richards_wolf_focus` — Richards-Wolf high-NA vector focusing.
- `debye_wolf_psf` — Debye-Wolf PSF computation.

#### Partial coherence (`coherence.py`, new module)
- `koehler_image` — Koehler illumination imaging.
- `extended_source_image` — extended-source incoherent imaging.
- `mutual_coherence` — mutual coherence function.

#### Detector model (`detector.py`, new module)
- `apply_detector` — shot noise, read noise, dark current, QE,
  full-well clipping, pixel binning.
- `shack_hartmann` — Shack-Hartmann wavefront sensor simulation.

#### Thin-film coatings (`coatings.py`, new module)
- `coating_reflectance` — transfer-matrix method (TMM) for multilayer
  dielectric coatings: R, T, phase vs wavelength and angle.
- `quarter_wave_ar` — single-layer AR coating designer.
- `broadband_ar_v_coat` — two-layer V-coat AR designer.

#### Interferometry (`interferometry.py`, new module)
- `simulate_interferogram` — generate fringe patterns from OPD maps.
- `phase_shift_extract` — 4-step phase-shifting interferometry.
- `fringe_spacing` — fringe spacing calculator.

#### Freeform surfaces (`freeform.py`, new module)
- `surface_sag_xy_polynomial` — XY polynomial departure from base conic.
- `surface_sag_zernike_freeform` — Zernike polynomial freeform.
- `surface_sag_chebyshev` — Chebyshev polynomial freeform.
- `surface_sag_freeform` — unified dispatcher.

#### Ghost analysis (`ghost.py`, new module)
- `enumerate_ghost_paths` — find all double-bounce ghost paths.
- `ghost_analysis` — trace ghost paths and compute intensity.

#### RCWA (`rcwa.py`, new module)
- `rcwa_1d` — rigorous coupled-wave analysis for 1D gratings.
- `grating_efficiency_vs_wavelength` — spectral efficiency sweep.

#### Multi-configuration (`multiconfig.py`, new module)
- `Configuration` dataclass for multi-config merit evaluation.
- `multi_config_merit` — weighted merit across configurations.
- `create_zoom_configs` — zoom-system configuration builder.
- `afocal_angular_magnification` — angular magnification from ABCD.
- `beam_expander_prescription` — Galilean beam expander builder.
- `keplerian_telescope` — Keplerian telescope builder.

#### Sources (`sources.py`, expanded)
- `create_tilted_plane_wave`, `create_point_source`.
- `create_multi_field_sources` — multi-field-angle source array.
- `create_top_hat_beam`, `create_annular_beam`.
- `create_fiber_mode` — LP01 fiber mode.
- `create_led_source` — incoherent LED model.
- `create_bessel_beam` — non-diffracting Bessel beam.

#### Phase retrieval (`phase_retrieval.py`, new module)
- `gerchberg_saxton` — Gerchberg-Saxton algorithm.
- `error_reduction` — error-reduction algorithm.
- `hybrid_input_output` — hybrid input-output (HIO) algorithm.

### Changed

- **SciPy FFT default** — `USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1`.
  All wave-propagation functions now multithreaded by default (2-4x speedup).
- **`slant_correction` default reverted to `False`** — empirical validation
  showed the paraxial formula is equal-or-better for most cases because ASM
  propagation between surfaces already encodes obliquity.
- `apply_real_lens` gains `seidel_correction` option (opt-in, off by
  default) for analytic higher-order correction on doublets.

### Fixed

- **Exit-vertex OPL correction** — `apply_real_lens_traced` now transfers
  rays from the last surface's sag to the flat exit vertex plane using the
  signed parametric distance.  Previously, off-axis rays ended at
  `z = sag(h) != 0`, injecting systematic defocus (43% on doublets) or
  catastrophic sign errors (200,000x on negative meniscus lenses with
  convex rear surfaces).  Doublet focus error: 10 mm to 0.000 mm.
  Negative meniscus residual: 33,742 nm to 0.17 nm.
- **Raytrace OPL bookkeeping** — `_intersect_surface` now accumulates
  `n_medium * t` for the vertex-to-sag leg.  Previously the ray moved to
  the sag intersection without counting that path.  Singlet residuals
  dropped 17x-130x.
- **TMM coating formula** — corrected B,C matrix extraction:
  `B = M[0,0] + M[0,1]*eta_sub` (was transposed).  Quarter-wave AR now
  gives R = 0 exactly at the design wavelength.

### Validation

- `validation/real_lens_opd/` — 21-case OPD validation suite comparing
  three methods (paraxial, slant-corrected, ray-traced) against geometric
  truth.  All cases show sub-nm traced RMS.  Matching Zemax LDE + .zmx
  exports for cross-verification.

---

## [2.5.0] — Prior release

- Core ASM/Fresnel/Fraunhofer propagation.
- `apply_real_lens` thin-element phase-screen model.
- Geometric ray tracer with ABCD, spot diagrams, Seidel coefficients.
- Glass catalog (Sellmeier, refractiveindex.info integration).
- Gaussian, Hermite-Gauss, Laguerre-Gauss source models.
- Polarization (Jones calculus).
- DOE / microlens array generation.
- HDF5 field I/O.
- Plotting utilities.

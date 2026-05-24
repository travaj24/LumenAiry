# LumenAiry Forward Roadmap

**Last updated:** 2026-05-21 (post-v5.2.3).  The entire v5.0 / v5.1
/ v5.2.x ROADMAP has shipped:

* **v5.0** -- coordinated breaking-change release (Python 3.10
  floor, 5 shim removals, `system.py` move, CI gates,
  Migration-Guide.md).
* **v5.1.0** -- 6 large-file splits (~26K LOC reorganised into
  ~35 submodules; public API bit-for-bit preserved) +
  library-wide default-knob consumer wiring.
* **v5.2.0** -- 4 new meta-walkers (V12 / V13 / V14 / V15), all
  v5.x deferred features (MCF / formula-3 evaluator / off-axis
  conic / 5 examples / 57-file consolidation / Chebyshev
  extraction), structural cleanups (_xp_of dedup / backend-fft
  inversion fix), CONVENTIONS table + validation/README, 5
  v4.13.1-deferred physics fixes (DOE sign / scale_floor /
  output_grid rename / MHS guard / partition-of-unity warning),
  mypy CI activation, 85% ruff cleanup.
* **v5.2.1** -- complete ruff baseline closure (134 -> 0) +
  numexpr `local_dict=` refactor.
* **v5.2.2** -- Python 3.10 install-path fix (pyfftw 0.15.1 +
  zarr 3.x dropped 3.10).
* **v5.2.3** -- Tier-1 / Tier-2 / Tier-3 ROADMAP closure: 24
  formula-3 coefficients ingested, MHS substantive resampling,
  subaperture image-plane mapping, `ao_closed_loop` helper,
  V16 walker + verify_changelog_closures script + dep-drift
  weekly cron, 9 source-text-proxy tests behavioral conversion,
  README cookbook split, partial CHANGELOG archive.
* **v5.2.5** -- AUDIT_V5_2_3 closure (7 P2 + 10 P3): V12 regex
  + V16 heading-classifier widening (the walker bugs the v5.2.3
  release itself surfaced), publish.yml verify matrix gets
  Python 3.10, HFPI/HF freespace dispatcher threads
  `output_grid`/`output_dx`, AST `_resolve_arg_closure`
  function-scope + last-write-wins tightening, ao_closed_loop
  `leak` + `tol` kwargs + edge-case handling, Chebyshev
  derivative + second-derivative `xp=` dispatch, Example 09
  Strehl normalization (library + example side), `_install_atexit_restore`
  rename + alias, cookbook cross-links, V15 floor bumped 5 -> 6.
  3781 unit tests pass (force-retagged post-tag for CI checkout
  fix).
* **v5.3.0** -- AUDIT_V5_2_5 closure (1 P1 + 2 P2 + 12 P3) PLUS
  all 3 remaining v5.x ROADMAP horizon items.  HF freespace
  TypeError regression fixed (the v5.2.5 P2-F1-1 closure shipped
  half-broken); V17 walker added for recursive self-citation
  drift; ao_closed_loop docstring honesty; AST tightening
  against 4 new bypasses; MultiFieldMerit JIT (8.1x speedup at
  N=256/8 fields); CHANGELOG.md pre-v4.11 archive complete;
  pyproject conftest comment corrected; ROADMAP code-work is
  now empty.  3848 unit tests pass.
* **v5.3.1** -- docs-only: strip stale ``What's new`` block from
  README; PyPI project-page now points to CHANGELOG.md
  canonical source.
* **v5.3.2** -- ships the 3 remaining v5.x horizon items as a
  package:
  - **Per-iteration logging telemetry** on the 3 named long-
    running paths (``apply_real_lens_traced``, ``design_optimize``,
    ``monte_carlo_tolerancing``).  ``lumenairy/_logging.py``
    NullHandler-default convention.  Existing 42 ``warnings.warn``
    sites stay -- TELEMETRY scope, not warning-conversion.
  - **V18 walker** for source-file:line citation drift
    (``tests/unit/test_v5_3_2_walker_source_line_citation.py``)
    + companion ``scripts/check_source_line_citations.py``
    wired into publish.yml's verify job at line 93.  Catches
    the v5.3.0-class ``wrapper_merits.py:855 -> :876``
    drift pattern in general.
  - **CHANGELOG ship-time-stamp script**
    (``scripts/stamp_changelog.py``) + ``docs/release-process.md``
    documenting the four invocation patterns.  Closes the
    V17-detected recursive self-citation drift class.

After v5.3.2: the v5.x ROADMAP **library code-work** is fully
closed.  The v5.3.2 cycle generated 2 follow-up audits which
v5.4.0 then closed:

* **v5.4.0** -- closes both AUDIT_V5_3_2_2026_05_23 (1 P2 physics
  + 5 P2 walker + 10 P3 = 16 items) and
  AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 (6 Tier-1 P1 + 5 Tier-2
  P2 + 3 Tier-3 P3 = 14 GUI items) in a single coordinated ship.
  Library: HF freespace Parseval renorm, V17/V18 walker
  hardening (5 narrowing surfaces closed), P3 code/doc batch.
  Designer GUI: 6 new docks (wavefront map, AO closed-loop,
  coronagraph workflow, operator algebra, thin-film coatings,
  log viewer, Chebyshev fit) + 5 dock expansions (optimizer
  parameter surface, phase retrieval, ghost path enumeration,
  Stokes tab, coherence 4-tab).  Cross-cutting CancellableProgress
  + Stop buttons.  Designer GUI version bumped 3.7.10 -> 5.4.0
  (co-versioned with library).

After v5.4.0: the v5.x ROADMAP is **fully closed**, both library
AND Designer GUI.  Remaining horizon is process-only:
* Next audit cycle (AUDIT_V5_4_0_*) -- your call on cadence.
* Force-retag discipline retrospective -- v5.2.5, v5.3.0, and
  v5.3.2 each needed at least one post-tag commit for CI-
  environment-only issues; v5.3.1 was the only clean single-commit
  release in the cycle (zero post-tag commits).  Either accept
  the class and document it, or add a structural pre-tag check
  that runs ``ruff`` + ``check_source_line_citations`` + a
  synthetic fresh-clone smoke before pushing the tag.  See
  ``docs/audits/AUDIT_V5_3_2_2026_05_23.md`` Part 5 for the
  trajectory analysis.

This file captures the next-release scope for LumenAiry and its
Designer GUI.  Items are grouped by release target and prioritised
within each.  Each entry has a short rationale + scope estimate so
a future plan-phase agent dispatch can lift items directly into a
work plan.

Historical per-module limitation notes from the v4.9.0-era ROADMAP
are preserved in git history; this file is forward-only.

---

## Current state

- **Library:** v5.3.0 baseline (3848 unit tests passing + 17
  documented skips + 1 documented xfail = 3866 collected).
  Python 3.10+ required.
  v4.16.0 mega-rollup
  shipped the entire v4.16 + v4.17 + v4.18 ROADMAP in one release.
  v4.16.1 closes the v4.16.0 deep audit through P3: 4 silent-wrong-
  answer correctness bugs (`MultiWavelengthMerit` SUM→AVG,
  `shack_hartmann` pitch quantisation, `_detect_backend` directory
  misclassification, LM `bounds` None-endpoint); the Schell-model
  partial-coherence cluster (new `propagate_ensemble(...)` helper +
  retired default factory `DeprecationWarning` + MCF rejection
  message refresh); JAX-traceable dtype probe + high-NA UserWarning;
  10th cache-registry meta-pin walker; glass / compat / UX cleanup
  (`refractiveindex` moved to optional `[glass]` extras, `zarr>=3.0`
  floor, 4 stale `n_d` inline comments fixed, `ProcessPoolExecutor`
  spawn-context, new `examples/06_schell_propagation.py` +
  `examples/07_zemax_load_trace.py`, new `CONVENTIONS.md`).  34/34
  validation files passing.  Public API at ~400+ symbols in
  `lumenairy.__all__`.
- **Designer GUI:** v5.4.0 (co-versioned with the library; reads
  `lumenairy.__version__` at runtime).  37 docks: 22 pre-v5.4 +
  6 NEW in v5.4.0 (wavefront map, AO closed-loop, coronagraph
  workflow, operator algebra, thin-film coatings, log viewer,
  Chebyshev fit) + 5 substantive expansions (optimizer parameter
  surface, phase retrieval, ghost path enumeration, Stokes tab,
  coherence 4-tab).  See `docs/designer_guide.md` for the full
  dock-by-dock library-backing inventory.
- **Audit closure status:** AUDIT_V4_12_1 through
  AUDIT_V4_16_3 all closed; AUDIT_V5_0_0 in active v5.0.1 closure.
  AUDIT_V4_13_1 Tier-2/3/4 architectural items now scoped to
  v5.1+ (the v5.0 release shipped breaking changes only; the
  structural reorganisation follows in v5.1).
- **Active back-compat shims (post-v5.0):** 3 -- the 5 shims
  catalogued in AUDIT_V4_13_1 Part 5 that v5.0 removed
  (`analysis.analysis`, `ao`, `io.hdf5`, JAX aperture legacy
  schema, `cosmic_ray_rate` kwarg) are gone; the 3 remaining are
  `lumenairy.elements.lenses.apply_*_lens` re-exports
  (intentionally kept as legitimate one-stop public API surface
  per the v5.0 CHANGELOG "Shims preserved" decision -- a v5.2+
  audit that flags these for removal should be rejected with a
  pointer to the v5.0 CHANGELOG entry).
- **Meta-pin coverage:** ALL 18 dispatcher meta-pins active and
  clean (V12-V15 added in v5.2; V16 in v5.2.3; V17 in v5.3.0; V18
  in v5.3.2; see the "v5.2-class sibling-gap surfaces" note below):
  - V1: cache-clear chain re-export (v4.14.1).
  - V2: cache <-> lock pairing (v4.14.2).
  - V3: 0+0j literal sweep (v4.14.2).
  - V4: input-validation entry-point (`_validate_grid_params`,
    v4.15.0).
  - V5: `_check_2d_scalar_field` 2-D-scalar-field guard (v4.15.3
    + scope extension in v4.15.4 + V6 first-positional-param
    discovery in v4.15.5).
  - **V6 (NEW v4.16.0)**: sentinel-aware branch propagation
    walker.
  - **V7 (NEW v4.16.0)**: `_xp_of` cross-backend dispatch
    walker.
  - **V8 (NEW v4.16.0)**: `dy` parameter threading walker for
    `apply_*` in `__all__`.
  - **V9 (NEW v4.16.0)**: `__all__` symmetry walker (submodule
    `__all__` must be re-exported at top level OR marked
    `_INTERNAL`).
  - **V10 (NEW v4.16.1)**: `@lru_cache <-> _cache_registry`
    enrollment walker -- AST-walks every `@lru_cache`-decorated
    module-level function and asserts a paired
    `_cache_registry` enrollment.  15 caches, 8 enrolled, 7
    exempt, 0 orphans.  v4.16.2 hardened to require the
    enrollment call appear at module level (not nested inside
    a function or `if False:` branch).
  - **V11 (NEW v4.16.2)**: doc-consistency walker -- scans
    `README.md`, `requirements.txt`, `ROADMAP.md` and
    `CHANGELOG.md` against `pyproject.toml`'s
    `[project.dependencies]` / `[project.optional-dependencies]`
    for dependency-declaration drift.  Closes the
    v4.16.1-identified documentation-surface sibling-gap pattern.
  - **V12 (NEW v5.2)**: CHANGELOG-vs-changeset walker -- parses
    the most-recent ``## [X.Y.Z]`` CHANGELOG block, asserts every
    backticked file-path citation resolves, every audit-ID
    citation resolves to ``docs/audits/``, the test-count
    arithmetic reconciles, and the block advertises an audit-
    closure verification mechanism.  Closes the v5.1.0-identified
    CHANGELOG-vs-implementation sibling-gap pattern (the
    "fabrication" class) at the file-existence + audit-ID level.
    Content-level fabrications still require human review or the
    companion ``scripts/verify_changelog_closures.py`` (v5.2+).
  - **V13 (NEW v5.2)**: shell-vs-canonical-location walker --
    for every name imported in a post-v5.1 file-split shell's
    ``from .X import Y`` block, asserts ``Y.__module__`` is the
    submodule (not the shell).  Catches the regression where a
    function body silently moves back into the shell.  Documented
    exemptions: ``propagate_modal_asymptotic`` (v4.14.1 monkey-
    patch contract).
  - **V14 (NEW v5.2)**: PEP-562 forwarding completeness walker --
    enumerates ``fft_infra`` mutable globals (those rebound via
    ``X = ...`` somewhere other than initial definition) and
    asserts each appears in ``propagation._LIVE_FORWARD_NAMES``.
    Counter-pin verifies the whitelist hasn't drifted in the
    opposite direction (names removed from ``fft_infra``).
    Catches the ``_PYFFTW_BAD_SHAPES``-class stale-snapshot
    regression that v5.1.1 closed.
  - **V15 (NEW v5.2)**: sentinel ``__reduce__`` structural
    walker -- auto-discovers every ``_Sentinel`` subclass via
    ``__subclasses__()``, asserts each defines ``__reduce__`` ->
    ``(_sentinel_unpickle, (name,))`` with the name registered in
    ``_SENTINEL_REGISTRY``, and verifies pickle round-trip
    identity.  Replaces the v4.15.2 hardcoded
    ``EXPECTED_SUBCLASSES`` tuple so new sentinels are auto-
    pinned (closes the v5.1.0 P3-NEW-F1-3 counter-pin gap).
  - **V16 (NEW v5.2.3)**: content-level CHANGELOG fabrication
    walker -- extends V12's file-existence + audit-ID coverage
    by parsing each audit-closure bullet and verifying via
    ``git diff PREV_TAG..HEAD`` that the claimed behavior change
    actually appears in the changeset.  Companion CLI script
    ``scripts/verify_changelog_closures.py`` (547 LOC) wired
    into ``publish.yml``'s verify gate.  Closes the v5.2.0-
    surfaced "cited file exists but cited behavior is missing"
    fabrication class.
  - **V17 (NEW v5.3.0)**: recursive self-citation drift walker
    -- pins CHANGELOG numeric self-citations (test count, file
    count, line count) against ``pytest --collect-only`` + ``git
    diff PREV_TAG..HEAD`` + ``wc -l CHANGELOG.md`` with a +/- 5
    drift tolerance.  Catches the v5.2.5-surfaced class where
    each entry's at-write-time empirical numbers diverged from
    at-ship-time reality.  v5.3.2 ships the ``stamp_changelog.py``
    pre-tag hook that closes the drift class structurally.
  - **V18 (NEW v5.3.2)**: source-file:line citation drift
    walker -- parses the topmost CHANGELOG block, extracts ALL
    backticked ``lumenairy/foo/bar.py:LINE`` (or ``:START-END``)
    citations, opens each at the cited line, and verifies the
    line is non-trivial.  The GENERAL version of
    ``test_v4_15_agent_f.py::TestF5ChangelogLineCitations``
    (which only catches a hardcoded shortlist of symbols).
    Companion CLI script ``scripts/check_source_line_citations.py``
    (429 LOC) wired into ``publish.yml`` BEFORE the V16 step.
    Catches the v5.3.0-surfaced ``wrapper_merits.py:855 -> :876``
    drift pattern in general.

  **v5.2 meta-pattern note**: At v4.16.2 the "fix N, miss N+1"
  sibling-gap meta-pattern was claimed retired across all
  currently-known surfaces.  v5.1.0 surfaced a NEW class
  (CHANGELOG-vs-implementation fabrication) that V11 did not
  cover; v5.2 closes it at the file-existence + audit-ID
  level via V12, plus 3 additional structural surfaces (V13
  shell-vs-canonical, V14 PEP-562 forwarding, V15 sentinel
  __reduce__) that v5.1.0's audit identified as remaining gaps.
  Honest current status: structurally retired across 18 currently-known sibling-gap surfaces; new classes will
  continue to surface and be added to the V-walker family as
  identified, including CONTENT-LEVEL CHANGELOG fabrications
  (where the cited file exists but the cited behavior is
  missing) which V12 deliberately does NOT cover -- those need
  the diff-aware companion script + human review.

---

## v4.16.0 + v4.17.0 + v4.18.0 — ALL SHIPPED in v4.16.0 mega-rollup

The 9 items previously catalogued under v4.16 (2) + v4.17 (4) +
v4.18 (3) all shipped in the v4.16.0 mega-release.  See the
"Shipped highlights" section below for the per-item summary.
Quick recap:

* **v4.16 items (2)**: V4 meta-pin walkers (4 walkers — sentinel
  propagation, `_xp_of` dispatch, `dy` threading, `__all__`
  symmetry) + multi-process atomic-append for `storage.py` (HDF5
  SWMR + filelock Zarr lock).
* **v4.17 items (4)**: Constrained optimisation
  (`Constraint` dataclass + scipy `NonlinearConstraint`);
  checkpoint/resume (`state_file=` + atomic JSON write);
  multi-objective Pareto via pymoo NSGA-II (optional
  `multi_objective` extras group); Hessian / Newton-step
  (`method='newton'` + scipy `trust-ncg`).
* **v4.18 items (3)**: CDGM + Hikari + Sumita Sellmeier
  catalogues (32 new glasses; `GLASS_REGISTRY` 46 → 78);
  per-glass Sellmeier validity ranges (`GLASS_VALIDITY`);
  central cache registry (`register_cache_clearer` + 9 caches
  migrated; retires the lazy-import fan-out).

The next horizon is v5.0 — major structural release.

### Known issues flagged for v4.16.x

* **Bundled Sellmeier formula-3 (polynomial) evaluator** (NOT
  addressed in v4.16.1; deferred to v5.0).  Hikari, Sumita, and
  4 CDGM glasses use refractiveindex.info formula 3.  The bundled
  `_sellmeier_index` still only supports formula 2.  v4.16.1
  reduced the user-facing impact by moving `refractiveindex` to the
  optional `[glass]` extras group (so a minimal install no longer
  pretends to support these 26 entries — users who want formula-3
  glasses pip-install the extras).  A native formula-3 evaluator
  in the bundle is a v5.0 candidate.

---

## v5.1, v5.2 -- shipped (see "Shipped highlights")

Everything that was in this section as of v5.0 has now shipped
across v5.1.0, v5.2.0, v5.2.1, v5.2.2, and v5.2.3.  Brief recap
of the v5.1+v5.2 shipped items (the "Shipped highlights" section
near the end of this file carries the per-release detail):

* **6 file splits** -- v5.1.0.  26K LOC reorganised into ~35
  submodules; public API bit-for-bit preserved via re-export shells.
* **Library-wide default-knob consumer wiring** -- v5.1.0.
  `set_default_wave_propagator` / `set_default_dy` /
  `set_default_real_dtype` now consulted by every applicable
  entry point.
* **57-file `test_audit_fixes_*` consolidation** -- v5.2.0.
  791 tests preserved into 10 topical homes.
* **Shared Chebyshev helpers extraction** -- v5.2.0.
  `lumenairy/_math/chebyshev.py` is the canonical home; 6
  consumer sites updated, back-compat aliases preserved.
* **MCF top-level alias** -- v5.2.0.  `lumenairy.MCF` is now
  `PartialCoherenceMCF`.
* **26 / 24 formula-3 glass coefficients** -- evaluator
  shipped at v5.2.0; **24 coefficient sets ingested at v5.2.3**
  from the refractiveindex.info database with worst-case n_d
  delta 7.9e-6 (under the 5e-5 budget; no tolerance relaxation
  required).  ("26" was the original ROADMAP count; the
  catalogue actually has 24 formula-3 polynomial entries -- 4
  CDGM + 10 Hikari + 10 Sumita -- so v5.2.0 auto-corrected the
  manifest count.)
* **Off-axis conic in surface frame** -- v5.2.0.
  `apply_real_lens(..., surface_frame=True)` opt-in with full
  rigid-body transform.  Default `False` preserves v5.1
  behavior bit-for-bit.
* **5 new examples** (multiconfig zoom / Monte-Carlo tolerancing
  / coronagraph / AO closed loop / ghost stray-light) -- v5.2.0.
  Example 11 (AO closed loop) re-built on the v5.2.3
  `ao_closed_loop` high-level helper.
* **CONVENTIONS.md sign-convention table** -- v5.2.0.
* **validation/README.md** -- v5.2.0.
* **README.md split** (deep cookbook -> `docs/cookbook.md`) --
  v5.2.3.
* **CHANGELOG.md archive split** (v4.11-v4.12 entries ->
  `docs/changelogs/v4.md`) -- v5.2.3 (partial; pre-v4.11
  entries still in top-level CHANGELOG.md, deferred to v5.3
  for completion).
* **mypy strict CI activation** -- v5.2.0.  All 76 scope-local
  errors cleaned; `mypy` is now a real CI gate
  (`continue-on-error: false`).
* **Ruff baseline cleanup** -- v5.2.0 (917 -> 134, 85%) +
  v5.2.1 (134 -> 0).  `lint` job now finishes green.

### Active back-compat shims at v5.0 (intentionally kept)

v5.0 removed 5 of the 8 shims catalogued in AUDIT_V4_13_1
Part 5.  The 3 remaining shims are **intentionally retained**
as legitimate public API surface, NOT scheduled for removal:

* `lumenairy.elements.lenses.apply_thin_lens` re-export.
* `lumenairy.elements.lenses.apply_real_lens` re-export.
* `lumenairy.elements.lenses.apply_real_lens_traced` re-export.

These give a coherent one-stop ``from
lumenairy.elements.lenses import apply_real_lens`` import
surface; the underlying `_lens_thin.py` / `_lens_real.py` /
`_lens_traced.py` file-split is an internal organisational
choice rather than a deprecation cycle.

---

## v5.3 forward horizon

v5.2.3 closed every Tier-1 / Tier-2 / Tier-3 item that the
v5.2.0 + v5.2.1 + v5.2.2 patches had deferred to v5.3.  The
items that remain open at v5.2.3 ship are:

### Performance (no concrete cost / value triggers yet)

* **`MultiFieldMerit` JIT compile**.  Per-field `np.exp` +
  `np.where` calls dominate at large N; meshgrid cache hits
  1.19x at N=128.  Numba or JAX JIT compile of the per-field
  tilt path would lift the ceiling but adds an optional-dep
  hot path.  Not blocking any current user.

* **`logging` adoption sweep**.  42 `warnings.warn` calls
  across 22 files; long-running paths (`apply_real_lens_traced`,
  `design_optimize`, `monte_carlo_tolerancing`) have no
  per-iteration telemetry.  Convention change spanning the
  whole library; cost is high vs the immediate need.

### CHANGELOG archive completion

* The v5.2.3 archive split moved v4.11 - v4.12 entries into
  `docs/changelogs/v4.md`.  v4.10 and earlier (down to v2.5)
  are still in the top-level `CHANGELOG.md`.  v5.3 finishes
  the archive pass.

### Designer GUI horizon -- ALL SHIPPED in v5.4.0

The 6 unscoped Designer items previously listed here all shipped
in v5.4.0 (driven by AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24).  See
`docs/designer_guide.md` for the v5.4.0 dock inventory.

* **Polarization plotting docks** -- SHIPPED.  Jones pupil was
  already shipped at v3.6+; v5.4 added Stokes + DOP/DOLP/DOCP tabs
  to `jones_pupil_dock.py` (193 -> 404 LOC).
* **Coronagraph workflow dock** -- SHIPPED at v5.4.  New
  `coronagraph_dock.py` (783 LOC) with the 4-stop chain builder.
* **Tolerancing dock** -- already shipped pre-v5.4 (`tolerance_dock.py`
  at 524 LOC; ROADMAP's prior "missing" claim was a self-citation
  drift caught by audit).  v5.4 adds Stop-button cancellation.
* **Multi-config / zoom dock** -- already shipped pre-v5.4
  (`multiconfig_dock.py` at 245 LOC; ROADMAP's prior "missing"
  claim was a self-citation drift caught by audit).  v5.4 adds
  Stop-button cancellation.
* **Wavefront-map plot integration** -- SHIPPED at v5.4.  New
  `wavefront_map_dock.py` (661 LOC) wrapping `plot_wavefront()`
  with embedded canvas + 6 controls + live optimiser hook.
* **`CancellableProgress` Stop button** -- SHIPPED at v5.4.  Wired
  in optimizer + tolerance + phase retrieval + multiconfig docks
  (6 worker classes total).

### Audit-cadence follow-ups

* **Audit cycle continuation**.  v5.2.x closed every
  v5.x-vintage audit P1-P3 finding.  v5.3 begins on the
  AUDIT_V5_2_X audit cycle (when written).  The 16
  meta-walkers V1-V16 + the dep-drift cron + the V12 /
  verify_changelog_closures content-vs-changeset gate cover
  the currently-known sibling-gap surfaces; new classes will
  continue to be added to the V-walker family as identified.

### Active back-compat shims at v5.0 (intentionally kept)

v5.0 removed 5 of the 8 shims catalogued in AUDIT_V4_13_1
Part 5.  The 3 remaining shims are **intentionally retained**
as legitimate public API surface, NOT scheduled for removal:

* `lumenairy.elements.lenses.apply_thin_lens` re-export.
* `lumenairy.elements.lenses.apply_real_lens` re-export.
* `lumenairy.elements.lenses.apply_real_lens_traced` re-export.

These give a coherent one-stop ``from
lumenairy.elements.lenses import apply_real_lens`` import
surface; the underlying `_lens_thin.py` / `_lens_real.py` /
`_lens_traced.py` file-split is an internal organisational
choice rather than a deprecation cycle.  A v5.2+ audit that
flags them for removal should be rejected with a pointer to
the v5.0 CHANGELOG "Shims preserved" decision.

---

## Designer GUI (co-versioned with library at v5.4.0+)

The Designer ships co-versioned inside the library wheel.  As of
v5.4.0 it reads `lumenairy.__version__` at runtime, so the v3.7.10
internal version markers are no longer authoritative.  See
`docs/designer_guide.md` for the v5.4.0 dock surface inventory.

v5.4.0 ships 6 new docks + 5 expansions per
AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24.  No open v5.x ROADMAP
items remaining for the Designer.

Possible v5.5+ scope (not yet planned):

* **Polarization plotting docks** — none currently surface Jones-
  pupil and Stokes maps from the library's `polarization.py` /
  Richards-Wolf paths.
* **Coronagraph workflow dock** — `analysis/coronagraph.py` has
  `coronagraph_contrast_curve` but no dedicated dock to set up
  the 4-stop chain (Lyot focal mask → Lyot stop → apodised pupil)
  interactively.
* **Tolerancing dock** — `monte_carlo_tolerancing` exists but the
  UI surface is limited; a dedicated "perturbation knobs + run MC"
  dock is the canonical Zemax pattern.
* **Multi-config / zoom dock** — `optimize/multiconfig.py` exists
  but no UI; users build configs in code.
* **Wavefront-map plot integration** — v4.14.0 added
  `plot_wavefront`; no dock surfaces it yet.
* **`CancellableProgress` UI button** — v4.13.1 added the
  cancellation protocol, wired into all 4 scipy callbacks; needs
  a Stop-button surface in the optimisation dock.

---

## Opportunistic / lower-priority items -- all shipped at v5.2.x

(Section kept as a historical pointer; every item shipped between
v5.2.0 and v5.2.3.  See "Shipped highlights" for per-release
detail and the "v5.3 forward horizon" section above for the two
remaining performance items.)

* `_deprecation.py` orphan helpers -- documented as kept-for-
  forward-use at v5.2.0.
* Duplicate `_xp_of` -- consolidated at v5.2.0 (5 sites -> 1
  `from ..backend import array_namespace as _xp_of` alias).
* `backend/fft.py -> propagators/propagation.py` inversion --
  fixed at v5.2.0 (now routes through `fft_infra` directly).
* Source-text-proxy tests (4+ sites) -- 5 replaced behaviorally
  at v5.2.3; 4 kept by design as anti-pattern absence checks.
* `output_grid` parameter semantics (AUDIT_V4_13_1 P1-A) --
  sub-propagator rename + `DeprecationWarning` shim at v5.2.0;
  dispatcher forwarding fix at v5.2.3.
* MHS subdomain grid loss (AUDIT_V4_13_1 P1-C) -- safe
  `ValueError` guard at v5.2.0; substantive maslov-branch
  resampling at v5.2.3 (Parseval-preserving, rel_err < 1e-9).
* Partition-of-unity convention (AUDIT_V4_13_1 P1-F) --
  `UserWarning` + opt-in kwargs at v5.2.0; full ABCD-driven
  image-plane mapping at v5.2.3 (unit-mag bit-for-bit
  preserved).
* `apply_doe_phase_traced` sign (AUDIT_V4_13_1 P1-G) -- fixed
  at v5.2.0.
* `MultiPrescriptionParameterization.scale_floor`
  (AUDIT_V4_13_1 P1-1) -- shipped at v5.2.0 with per-type
  default table.
* `MultiFieldMerit` JIT + `logging` adoption -- still open;
  moved to "v5.3 forward horizon" above.

---

## Recommended sequencing

(Historical pre-v5.0 sequencing: stack v4.16 -> v4.17 -> v4.18
as focused minors, then v5.0 as the coordinated breaking-change
release.  All four releases shipped; see "Shipped highlights"
below for the per-release summary.)

Post-v5.0 sequencing:

- **v5.0.1** -- patch closure of the AUDIT_V5_0_0 findings:
  3 P1s (lint baseline strategy + F821 forward-ref real bugs +
  `set_default_*` warning text honesty + `benchmarks/` import-
  path drift) + 5 P2s (asymmetric shim-removal anti-regression
  pin coverage; stale ROADMAP v5.1 section; `simulate_detector
  _image` doc-naming consistency; 2 stale docstrings; mypy
  config preparation) + 8 P3s (test-count arithmetic, doc
  drift, stale comments, MCF clarification, CI matrix vs
  classifiers, etc.).  No new features; no breaking changes.
- **v5.1** -- the structural reorganisation per the section
  above (6 file splits + Chebyshev extraction + 57-file test
  consolidation + library-wide default-knob consumer wiring
  + MCF public-API polish + formula-3 glass coefficients +
  off-axis conic + 5 missing examples + mypy CI activation
  + ruff-baseline cleanup PR).
- **v5.2+** -- the Opportunistic items above as scoped
  capacity allows; no breaking changes planned.

---

## Shipped highlights (since v4.9.0)

(Brief; the full per-release breakdown is in
[`CHANGELOG.md`](CHANGELOG.md).)

- **v4.10–v4.11.x** — comprehensive multi-agent physics audit
  response (~100+ findings).  Welford-mirror convention,
  C-LR-1 saga, raytrace + GBD + HF + subaperture + sources +
  detector closures, Sellmeier registry expansion.
- **v4.12.x** — Tier-1 perf wins (ASM 4.3×, jit caches 36-163×),
  pre-PyPI audit closure, cache hygiene infrastructure (7 LRU
  caches + `lumenairy_context(clear_caches_on_exit=True)`).
- **v4.13.0** — S1/S2/S3 + L2/L3/L4/L6/L8 audit closure,
  `except Exception:` sweep (99 → 3 sites), Tier-2 perf bundle
  (188× BSDF TIS, 10× thin-grating, 10.8× SH FFT batching,
  4-72× Seidel field sweep), `rcwa.py` → `thin_grating.py`
  rename.
- **v4.13.1** — AUDIT_V4_13_0 closure (3 sibling-gap P1s + 9 P2 +
  6 P3); 3 new perf wins (SH scatter 9.5-25×, vec-acc 1.65×,
  GBD reconstruct 1.2-1.5×); `CancellableProgress` cancellation
  protocol wired into 4 scipy callbacks.
- **v4.13.2** — AUDIT_V4_13_1 Tier-0 (12 P1s + 5 cross-survey P0s
  + thin-lens sibling sweep + latent CuPy dispatch bug).
- **v4.14.0** — AUDIT_V4_13_1 Phase B (7 Tier-1 perf wins
  including 24.6× coatings + 77× LG-mode cache + 6.17×
  Multi*Merit cache; 6 new public functions: encircled energy,
  MTF cutoff, beam diameter, depth of focus, plot_wavefront;
  80 parametrized dispatcher pins closing the sibling-gap
  audit-meta-finding).
- **v4.14.1** — AUDIT_V4_14_0 closure (1 P0 + 10 P1s including
  cache↔lock pairing meta-pin + LG mode-stack dx/dy correction
  + makedammann2d SI per-parameter heuristic + clear_asm_caches
  chain extension to 5 sibling caches).
- **v4.14.2** — AUDIT_V4_14_1 closure (1 P0 glass-registry +
  10 P1s + 2 new meta-pins: cache↔lock pairing
  (`test_v4_14_2_dispatcher_pin_cache_locks.py`, 39 tests; 38
  pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK` skip) +
  `0+0j` literal sweep
  (`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`, 123 tests).
  Doc reorganisation moved 10 audit `.md` files into
  `docs/audits/` and 7 release notes into `docs/release_notes/`.
- **v4.14.3** — AUDIT_V4_14_2 P0+P1 closure (2 P0s including
  storage `n_planes` atomicity + makedammann2d >1m upper-bound;
  5 sibling-gap P1s including LG-polynomial chain + apply_rotator
  conflict symmetry + create_led_source scale-inversion check;
  1 real physics error fix in multiconfig.py n=1.5 hardcoding).
  1265 unit tests; 34/34 validation.
- **v4.15.0** — AUDIT_V4_14_2 P2/P3 sweep + v4.15/v4.16 ROADMAP
  rollup.  Highlights:
  - **Meta-pin candidate #3**: input-validation entry-point
    pin (`test_v4_15_dispatcher_pin_validate_grid_params.py`,
    18 tests; 14 factories discovered with 1 documented
    `create_led_source` exemption -- legacy-shim positions
    validator past the 15-line head window).
  - **`lumenairy_context` redundant-call elimination** -- the
    `clear_caches_on_exit=True` exit path now issues a single
    `clear_asm_caches()` call rather than open-coding the
    7 sibling fan-out (eliminates 6+ redundant lock acquisitions
    per context-manager exit).
  - **HDF5/Zarr `lumenairy_version` attribute stamping** at every
    `create_dataset` / `create_array` site (storage.py).
  - **Source-factory validation completeness**:
    `create_multi_field_sources` now in `_validate_grid_params`
    call list (previously transitively validated via
    `create_tilted_plane_wave`; error message leaked internal name).
  - **Partial-coherence source trio** (originally v4.16 scope,
    shipped earlier in v4.14.x but only now folded into the
    ROADMAP shipped list): `create_gaussian_schell_source`,
    `create_schell_model_source`, `create_annular_incoherent_source`.
  - **6 v4.16-scope user-facing API items shipped early** (closes
    AUDIT_V4_13_1 cross-library-survey items #4-#8 + #10; v4.15.5
    Agent C moved these from the "v4.16 residual" ROADMAP section
    to Shipped highlights -- the duplicate-counting drift flagged
    in multiple recent audits):
    - `ee_polychromatic(rx, wavelengths, weights, radii, ...)` --
      polychromatic encircled-energy convenience helper chaining
      `polychromatic_psf` + `encircled_energy_radius`.
    - `strehl_vector(Ex, Ey, Ez=None, *, reference=None)` and
      `coupling_efficiency_vector(...)` -- polarisation-aware
      Strehl / coupling for Richards-Wolf / vector-imaging paths.
    - `rayleigh_resolution(psf, dx, wavelength, *, axis='radial')`,
      `sparrow_resolution(psf, dx)`, `fwhm_resolution(psf, dx)` --
      standard two-point separability definitions.
    - `astigmatism_mag_angle(coeffs)` -- Mahajan §8.2 conversion
      of `(c5, c3)` Zernike astigmatism to `(|astig|, theta)`.
    - `make_off_axis_parabola(focal_length, off_axis_angle,
      clear_aperture, ...)` -- OAP factory replacing manual
      tilt+decenter (v4.15.1 P0 fix corrected chief-ray launch to
      `2 f tan(alpha)`).
    - `surface_sag_q_bfs(X, Y, *, radius, coefficients, r_max, ...)`
      -- Forbes Q-bfs aspheric freeform (radial; the asymmetric
      2-D variant remains a v4.16+ deferral).
  - **CHANGELOG line-citation drift fix** (P3): refreshed
    `optimize/core.py:2750-2755` → `:2790-2795` and `:958-966` →
    `:977-991` to match the post-v4.14.2 drift, plus the
    `0+0j` literal-site citation `:966` → `:987`.
  - **README Cookbook section** added with runnable examples
    for the 6 v4.14.0 public functions + a `makedammann2d
    _legacy_units='SI'` migration example.
- **v4.15.1** — AUDIT_V4_15_0 closure + CLUSTER_B operator-algebra
  rollout.  Highlights:
  - **CLUSTER_B Item 2 — `lumenairy.algebra` operator algebra**:
    Nazarathy/Shamir-style symbolic optical-system construction
    (`Operator`, `CompositeOperator`, `FreeSpace`, `ThinLens`,
    `CylindricalLens`, `Magnify`, `FourierTransform`, `Aperture`,
    `GaussianAperture`) with closed-form 2x2 ABCD and chain-and-
    delegate field application onto the canonical LumenAiry
    propagators.
  - **CLUSTER_B Item 3 — `rays_from_field` bridge**: phase-ratio
    direction-cosine extractor that lifts a complex 2-D field
    into a packed ``Rays`` bundle for the geometric raytracer.
    Multiple placement modes (`'centroid'`, `'uniform'`, `'cdf'`)
    and three angle methods (`'phase_ratio'`, `'unwrap_gradient'`,
    `'autocorr'`).
  - **Partial-coherence redesign (P0-NEW-2)**: the 3 Schell
    factories now return raw ensembles by default
    (`return_kind='ensemble'`) and gain a `return_kind='mcf'`
    branch that produces a `PartialCoherenceMCF` object with
    Wolf-1982 coherent-mode decomposition for N > 64.
  - **Forbes Q surface dispatcher** (P1-F1-1 alignment): radial
    primary clip + rectangular secondary clip + dx threading.
  - **Sentinel consolidation (Agent E)**:
    `_ZeroApertureMaskSentinel` and `_AngleUnsetSentinel` now
    inherit from `_deprecation._Sentinel`; pickle round-trip
    preserves singleton identity via `_SENTINEL_REGISTRY` +
    `__reduce__`.
  - **`make_off_axis_parabola` P0 fix**: chief-ray launch radius
    corrected to `2 f tan(alpha)` (was `f tan(alpha)`, factor-of-2
    error at 30-deg surface-normal off-axis angle).
  - 1625 unit tests; 34/34 validation.
- **v4.15.2** — AUDIT_V4_15_1 P1/P2/P3 closure (placeholder; actual
  shipping summary populated at release commit time).  Highlights
  (Agent E scope): ROADMAP refresh to v4.15.1+ baseline; strict
  `_sentinel_unpickle` fallback (ImportError on unknown subclass);
  remaining `optimize/core.py` scalar-sentinel patterns promoted
  to `_Sentinel` subclasses for pickle safety;
  `PartialCoherenceMCF.coherence_at` Hermiticity unit test;
  end-to-end Forbes Q-bfs OPD analytical pin against
  `phi(r) = -k sag(r)`; `_NO_DEFAULT` upgraded to dedicated
  `_NoDefaultSentinel` for sentinel-class consistency; UI runtime
  test under `-W error::DeprecationWarning`; `Source.gaussian_schell`
  classmethod return-type aligned with the top-level factory
  (returns ensemble tuple instead of wrapping a 3-D ensemble in a
  `Source` whose `E` is 3-D); `lumenairy.algebra` exports moved
  from Tier-2 (Propagate) to Tier-1 (Build a system); sparrow
  tolerance pin tightened to <1% (achievable 0.02%) to match the
  analytical-value claim.
- **v4.15.3** — AUDIT_V4_15_2 closure with structural counter-
  measure (shared `_check_2d_scalar_field` helper + AST-walking
  dispatcher meta-pin V5).
- **v4.15.4** — AUDIT_V4_15_3 closure + meta-pin walker scope
  extension to `system.py` + `analysis/` + V5 broadening for
  `richards_wolf_focus`/`debye_wolf_psf`; `plot_opd_fan` +
  `plot_opd_summary` shipped.
- **v4.15.5** — AUDIT_V4_15_4 closure + V6 walker
  (first-positional-param-name discovery via AST inspection)
  closes the meta-pattern at the public-API surface; class-body
  descent with `_DELEGATING_CLASS_METHODS` exemption; 13 newly-
  guarded analyzers; `plot_opd_fan` `fan_units` kwarg +
  `_radial_rms_profile` centered RMS.
- **v4.16.0** — Mega-rollup: entire v4.16 + v4.17 + v4.18 ROADMAP
  shipped in one release.  Highlights:
  - **4 remaining V4 meta-pin walkers** land (sentinel
    propagation, `_xp_of` dispatch, `dy` threading, `__all__`
    symmetry).  Library now ships 9 active dispatcher meta-pins;
    sibling-gap meta-pattern at the public-API surface is
    structurally retired.  14 inline fixes across the 4 walker
    scans.
  - **Multi-process atomic-append for `storage.py`**: HDF5 SWMR
    mode + filelock-based distributed Zarr lock.  Subprocess
    multi-writer tests via `multiprocessing.get_context('spawn')`
    (Linux + Windows portable).  Single-process v4.14.3
    atomicity guarantees preserved bit-for-bit.  `filelock>=3.0`
    added to `hdf5` and `zarr` extras groups.
  - **Optimisation framework expansion**: `Constraint` dataclass
    + `design_optimize(constraints=...)` mapping to scipy
    `NonlinearConstraint`; checkpoint/resume via
    `state_file=` + atomic JSON write; `method='newton'`
    dispatching to scipy `trust-ncg` with FD-Hessian; multi-
    objective Pareto via pymoo NSGA-II wrapper
    (`design_optimize_multi_objective`) as optional
    `multi_objective` extras group.
  - **Glass + materials expansion**: 32 new glasses across CDGM
    (12) + Hikari (10) + Sumita (10); `GLASS_REGISTRY` 46 → 78;
    zero n_d cross-check failures at the 5e-5 tolerance.  Per-
    glass Sellmeier validity ranges (`GLASS_VALIDITY`, 77
    entries) with `UserWarning` on extrapolation.
  - **Central cache registry** (`lumenairy/_cache_registry.py`):
    `register_cache_clearer` + `list_registered_cache_clearers`
    + `clear_all_registered_caches`.  Retires the lazy-import
    fan-out in `clear_asm_caches`.  9 caches migrated.
    `clear_asm_caches` external contract preserved bit-for-bit.
    Counter-measure to the cache-clear "fix N, miss N+1"
    pattern.
  - 12 new top-level exports.  2106 unit tests pass (+184 net);
    34/34 validation files pass.

---

## How to update this file

When an item ships, move it from its release section to the
"Shipped highlights" section above with a one-line summary.  When
a new audit or cross-library survey adds items, append them under
the appropriate release target.  When v5.0 lands, archive this
file to `docs/roadmaps/ROADMAP_v4.md` and start fresh with the
v5.x forward plan.

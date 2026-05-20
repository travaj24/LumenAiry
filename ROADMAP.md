# LumenAiry Forward Roadmap

**Last updated:** 2026-05-19 (post-v4.16.1).  v4.16 + v4.17 + v4.18
ROADMAP scope all shipped in v4.16.0 mega-rollup; v4.16.1 closes the
v4.16.0 deep audit (4 silent-wrong-answer bugs + Schell/JAX cluster +
hygiene).  v5.0 is now the immediate horizon.

This file captures the next-release scope for LumenAiry and its
Designer GUI.  Items are grouped by release target and prioritised
within each.  Each entry has a short rationale + scope estimate so
a future plan-phase agent dispatch can lift items directly into a
work plan.

Historical per-module limitation notes from the v4.9.0-era ROADMAP
are preserved in git history; this file is forward-only.

---

## Current state

- **Library:** v4.16.1 baseline (2208 unit tests passing + 5
  documented skips + 1 documented xfail).  v4.16.0 mega-rollup
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
- **Designer GUI:** v3.7.10 (per in-code comments at
  `ui/main_window.py:2196` etc.).  No standalone release stream; the
  Designer ships co-versioned inside the library wheel.
- **Audit closure status:** AUDIT_V4_12_1 through
  AUDIT_V4_16_0_DEEP all closed.  AUDIT_V4_13_1 Tier-2/3/4
  architectural items scoped to v5.0 as noted below.
- **Active back-compat shims:** 8 (catalogued in AUDIT_V4_13_1 Part
  5).  v4.14.2 migrated 2 shims onto the canonical
  `_deprecation.warn_deprecated_signature` helper with explicit
  `version_removed=5.0`; the remaining 6 still use inline raw
  `warnings.warn` and are scheduled for migration as a v5.0
  housekeeping item.
- **Meta-pin coverage:** ALL 9 dispatcher meta-pins active and
  clean:
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
  The "fix N, miss N+1" sibling-gap meta-pattern at the public-API
  surface is now structurally retired across all known classes.

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

## v5.0.0 — Major structural release (breaking changes coordinated)

Held back from incremental releases because they're cross-cutting
and break public-API or test-organisation contracts.

### Architecture / housekeeping

* **6 file splits** (no public API change; mechanical reorganisation
  visible only to git-blame):
  - `raytrace/core.py` (4422 LOC) → split surface/intersection/
    trace/world-trace/Seidel/ray-fan/layout.
  - `propagators/propagation.py` (3710 LOC) → split FFT-infra/
    ASM/Fresnel/RS/SAS/MFT.
  - `propagators/asymptotic.py` (3597 LOC) → split modes/canonical-
    fit/aberration-tensor/Maslov/JAX-twin.
  - `optimize/core.py` (3258 LOC) → split parameterizations/merit-
    terms/wrapper-merits/context/driver/JAX-merits.
  - `io/prescriptions.py` (2829 LOC) → split builders/Zemax/CODE V/
    Quadoa/transforms.
  - `analysis/core.py` (2196 LOC) → split beam-stats/Strehl/PSF-
    MTF-OTF/polychromatic/Zernike/OPD.

* **CI gates** (currently absent):
  - `pytest tests/unit -m "not integration"` job (fast PR feedback).
  - `ruff check` lint gate.
  - `mypy --strict` incremental adoption starting with small
    modules (`backend/`, `_deprecation.py`, `_context.py`,
    `progress.py`, `memory.py`).
  - `tests/test_public_api.py` smoke test asserting every `__all__`
    entry is `getattr(lumenairy, name)`-resolvable.

* **Remove 8 active back-compat shims** (catalogued in
  AUDIT_V4_13_1 Part 5):
  - `analysis/analysis.py` (v4.7 rename shim)
  - `lumenairy/ao.py` (v4.3 shim)
  - `lumenairy/io/hdf5.py`
  - `elements/lenses.py:938-983` (v3.5.5 — 9 versions old)
  - `elements/elements.py:862-875` (v4.3)
  - `system.py:550-599` legacy aperture schema
  - `_deprecation.py` shims (v4.7)
  - `analysis/detector.py:288` deprecated `cosmic_ray_rate`

* **Shared Chebyshev helpers** extraction.  `propagators/asymptotic
  .py` and `elements/lenses_maslov.py` both import private
  Chebyshev helpers from `elements/lenses.py` — an inverted
  dependency.  Move to `lumenairy/_math/chebyshev.py`.

* **Audit-fix test-file consolidation**.  37 of 55 unit-test files
  are named `test_audit_fixes_v<X>_<Y>_<id>.py` (audit accretion).
  Merge into topical homes (`test_lens.py`, `test_propagation.py`,
  etc.); delete obsolete proxy tests (the `inspect.getsource`
  substring-match pattern caught in AUDIT_V4_13_1 Part 6.1).

* **`lumenairy/system.py` → `lumenairy/propagators/system.py`**
  (it walks elements applying propagators — functionally a
  propagator, not a top-level peer).

* **`__init__.py` reorganisation** so the import section's tier
  order matches `__all__`'s tier order (currently mismatched by
  ~5 tiers — see AUDIT_V4_13_1 Part 5 `__init__.py audit`).

* **Off-axis conic in surface frame** (not just decenter+tilt).
  `apply_real_lens` honours `decenter` and `tilt` keys but adds
  them to the field's coordinates, not the surface's frame.
  Tilted/displaced asphere with proper sag in surface frame
  requires a coordinate transformation Optiland and Zemax do
  natively.

* **Bump `requires-python` to >=3.10** (Python 3.9 EOL was
  2025-10).

### Config knobs

* `set_default_wave_propagator` — avoid passing
  `wave_propagator='gbd'` to every entry point.
* `set_default_real_dtype` — only complex has a knob today.
* `set_default_dy` — match v4.13.x anamorphic-threading direction.

### Documentation

* `validation/README.md` — currently absent; new contributors don't
  know whether to add tests to `tests/unit/` or `validation/`.
* `Migration-Guide.md` — currently missing; v4.13.0's `rcwa.py →
  thin_grating.py` rename and the `wavelength`-required-on-codegen
  are breaking changes that need migration prose.
* Convention table — one-stop summary of sign conventions
  (`exp(-i*omega*t)`, mirror radius wave-side `R>0=concave`,
  Welford signed-R for raytrace, OPD sign, etc.) — currently each
  is documented in the place it matters but no single doc page.
* Split README.md (3475 lines) — move deep cookbook to
  `docs/cookbook.md`.
* CHANGELOG.md (7000+ lines) — consider yearly / per-major-version
  archive splits.
* 5 missing examples: multi-config / zoom, tolerancing, coronagraph
  workflow, AO closed loop, ghost / stray-light.

---

## Designer GUI (separate version stream)

The Designer ships co-versioned inside the library wheel.  Current
in-code references: 3.7.6, 3.7.7, 3.7.9, 3.7.10 — latest is
**3.7.10**.  No active Designer plan since the prior `valiant-
hopping-dream` plan (3.6.1 + 3.7 + 3.8) was completed.

Possible v3.8+ scope (not yet planned):

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

## Opportunistic / lower-priority items

Catalogued from prior audits + cross-library survey, not allocated
to a specific release:

* **`_deprecation.py` orphan helpers** — `warn_deprecated_kwarg`,
  `warn_renamed_function`, `warn_deprecated_default` are exported
  but never called.  Either delete or wire into kwarg renames
  introduced in v4.7-v4.13.
* **Duplicate `_xp_of`** in `elements/elements.py:27` and `analysis
  /core.py:52`.  Consolidate to direct `from ..backend import
  array_namespace`.
* **`backend/fft.py` → `propagators/propagation.py` inversion** —
  FFT plan caches + toggle flags live in `propagation.py`, but
  `backend/fft.py` imports from there.  Lift to `propagators/
  fft_infra.py`.
* **Source-text-proxy tests** in v4.13.1's Agent 3 file — 4 tests
  use `inspect.getsource` substring matching instead of behavioural
  assertions.  Replace with real behaviour pins.
* **`MultiFieldMerit` performance ceiling** — meshgrid cache hit
  1.19× at N=128; the per-field `np.exp` + `np.where` calls
  dominate and cannot be cached.  Future: JIT-compile the per-
  field tilt path via Numba or JAX.
* **`output_grid` parameter semantics inconsistency** (AUDIT_V4_13_1
  Part 2 P1-A) — dispatcher documents `(N, dx_out)` but 3 sub-
  propagators (gbd/hfpi/hf) interpret as `(Ny, Nx)`.  Pick one;
  rename the sub-propagator kwarg to `output_shape` for the
  `(Ny, Nx)` case.
* **MHS subdomain grid loss** (AUDIT_V4_13_1 P1-C) —
  `prescription_subdomain` with default `method='maslov'` silently
  outputs on input grid (compounds dispatcher P1).
* **Partition-of-unity convention bug** in `subaperture.py:140-148`
  (AUDIT_V4_13_1 P1-F).  Window centred on source-plane positions;
  only correct for unit-mag no-tilt.
* **`apply_doe_phase_traced`** discards sign of `N` but `trace()`
  inline DOE kick preserves it (AUDIT_V4_13_1 P1-G).  ~2 LOC fix.
* **`MultiPrescriptionParameterization.scale_floor`** not supported
  (AUDIT_V4_13_1 P1-1).  ~20 LOC + path-aware defaults per opt
  variable type (radii / thicknesses 1e-6, conics 1e-3, aspheric
  α_n 1e-3/factor).
* **`logging` adoption** — currently 1 use across the whole library
  (`propagators/propagation.py:592`); 42 `warnings.warn` calls
  across 22 files.  Long-running paths (`apply_real_lens_traced`,
  `design_optimize`, `monte_carlo_tolerancing`) have no per-
  iteration telemetry.

---

## Recommended sequencing

Stack v4.16 → v4.17 → v4.18 as a sequence of focused minors
landing over weeks, then plan v5.0 as a coordinated breaking-
change release when the v4.16+ minor sequence stabilises.  v4.15
shipped the v4.14.2 P2/P3 sweep + the input-validation meta-pin
+ the partial-coherence source trio (pulled forward from v4.16).

- **v4.16** is a slim residual release after the v4.15.x sweep
  closed the user-facing API expansion: extend the meta-pin
  pattern with the 4 V4 candidates (sentinel propagation, `_xp_of`
  dispatch, `dy` threading, `__all__` symmetry) and finish the
  multi-process atomic-append story for `storage.py`.  v4.15.0 +
  v4.15.1 already shipped the 6 user-facing API items originally
  scoped here (polychromatic encircled energy, polarisation-aware
  Strehl, resolution metrics, astigmatism mag+angle, off-axis
  parabola, Forbes Q-type freeform); see Shipped highlights.
- **v4.17** is the optimisation-framework focus — constrained opt
  + checkpoint/resume + Newton-step.
- **v4.18** is the glass / materials expansion — broaden vendor
  coverage and tighten extrapolation safety + central cache
  registry to retire the lazy-import fan-out pattern in
  `clear_asm_caches`.
- **v5.0** is the major-structural release — file splits + CI
  gates + back-compat shim removal + Migration guide.  Coordinate
  timing with a Designer 4.0 if scope warrants.

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

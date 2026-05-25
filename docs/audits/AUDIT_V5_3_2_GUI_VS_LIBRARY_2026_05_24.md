# Audit -- Lumenairy Designer GUI vs Core Library at v5.3.2

**Date:** 2026-05-24
**Auditor:** Claude (3-agent parallel fleet: library-feature inventory + GUI-import scan + cross-reference gap matrix)
**Scope:** Audit the Designer GUI (`lumenairy/ui/`, v3.7.10, ~22,866 LOC across 45 files) against the post-v5.0 reorganised core library (v5.3.2). Two questions:
  1. Has the v5.0 -> v5.3.2 reorganisation broken any GUI -> library connections?
  2. Which new library features (v4.13.0 -> v5.3.2) have no GUI surface or under-exposed surfaces?

**Library tag under audit:** v5.3.2 (HEAD `b2c594d`)
**GUI version under audit:** 3.7.10 (per in-code references at `lumenairy/ui/main_window.py:2196` etc.)

**Verdict:** **GUI is hygienically connected to the post-v5.0 library** -- zero broken imports detected across 8 deprecated-pattern grep classes.  The actual gap is **new-feature wiring lag**: ~70 user-facing public API symbols shipped between v4.13.0 and v5.3.2; ~12 substantive features have no GUI surface or significantly under-exposed surfaces.  The ROADMAP's Designer-GUI section also contains two FALSE-NEGATIVE claims (it lists `tolerance_dock` and `multiconfig_dock` as missing when both are full implementations).

---

## Part 0 -- TL;DR

* **Broken connections:** none.  All 8 deprecated-pattern grep classes return 0 hits across the UI subpackage.  The v5.0 shim removals (`analysis.analysis`, `ao` legacy, `io.hdf5`, JAX aperture legacy schema, `cosmic_ray_rate` kwarg), the v5.1 file-splits (`_lens_thin` / `_lens_real` / `_lens_traced` private modules behind the `lumenairy.elements.lenses` shell), the v5.2.0 `output_grid` -> `output_shape` rename, the v5.2.5 `install_atexit_restore` privatisation, and the v5.3.0 HF freespace return-type change all left the GUI untouched.  The shell-re-export pattern + DeprecationWarning shims did their job.
* **Stale doc comments:** 11 docks carry v3.5-v3.8-era version markers in docstrings / inline comments.  Cosmetic only; no API impact.
* **ROADMAP corrections needed:** `tolerance_dock.py` (524 LOC, full Monte-Carlo perturbation UI) and `multiconfig_dock.py` (245 LOC, joint-optimisation across configs) both exist as complete implementations -- ROADMAP "missing GUI items" list still flags them.  `jones_pupil_dock.py` (193 LOC) covers the Jones-pupil half of the polarization-docks item; Stokes maps remain genuinely missing.
* **Actual gaps:** 6 P1 (user-blocking) + 5 P2 (significant) + 3 P3 (polish) = **14 GUI work items** to wire the v4.13.0 -> v5.3.2 library feature wave into the Designer.
* **Worker telemetry:** 10 QThread workers across 11 docks.  ZERO of them attach a handler to the `lumenairy` logger, so the v5.3.2 per-iteration telemetry (`apply_real_lens_traced`, `design_optimize`, `monte_carlo_tolerancing`) is silently consumed by the NullHandler.  ZERO of them wire `CancellableProgress` for a Stop-button surface (the library hook has been ready since v4.13.1).

This is a **clean library, half-wired GUI** -- the urgent work is wiring, not repair.

---

## Part 1 -- Methodology

Three parallel sub-agents (`Explore` class) ran simultaneously:

* **Stream A -- Library new-feature inventory.** Walked CHANGELOG.md + ROADMAP.md "Shipped highlights" + `lumenairy/__init__.py __all__` and enumerated every user-facing public-API symbol introduced between v4.13.0 and v5.3.2.  Categorised, dated, GUI-relevance scored.
* **Stream B -- GUI dock import + broken-connection scan.** Per-file LOC + docstring + lumenairy imports across all 45 `lumenairy/ui/*.py` files.  Exhaustive grep for 8 deprecated-pattern classes that v5.0 / v5.1 / v5.2 / v5.3 might have orphaned.  QThread worker enumeration + logging-handler / CancellableProgress wiring check.
* **Stream C -- Cross-reference + gap analysis.** Matched Stream A's library inventory against Stream B's dock surfaces to identify (a) library features without any dock, (b) docks that under-expose their backing library function's parameter / mode surface.  Verified ROADMAP "missing GUI items" against actual files.

This document synthesises the three streams.

---

## Part 2 -- Dock inventory (Stream B)

GUI total: **45 files, 22,866 LOC**.  Three tiers:

### 2.1 Core widgets (9 files, ~12,857 LOC -- 56% of UI)

| File | LOC | Description |
|------|-----|-------------|
| `main_window.py` | 3505 | Dock manager, menu bar, recent files, workspace tabs |
| `model.py` | 2843 | Central element-based system state; raytrace dispatch |
| `element_table.py` | 1305 | Element prescription editor (lenses, mirrors, DOEs) |
| `layout_2d.py` | 1054 | Interactive 2-D optical layout (QGraphicsScene) |
| `layout_3d.py` | 940 | 3-D VTK widget (pyvistaqt) |
| `workspace.py` | 647 | Workspace / tab system for dock grouping |
| `analysis.py` | 395 | Spot-diagram + ray-fan + system-summary widgets |
| `surface_editors.py` | 391 | Aspheric / biconic / freeform / coating coefficient editors |
| `surface_table.py` | 352 | QTableModel-backed surface spreadsheet |

### 2.2 Major analysis docks (13 files, ~6,231 LOC)

| File | LOC | Backing library calls |
|------|-----|------------------------|
| `waveoptics_dock.py` | 2714 | `angular_spectrum_propagate`, `fresnel_propagate`, `apply_real_lens_traced`, `resample_field` |
| `optimizer_dock.py` | 1022 | 3x QThread workers: `OptimizeWorker`, `WaveOptimizeWorker`, `GlobalSearchWorker` |
| `tolerance_dock.py` | 524 | `ToleranceWorker` (QThread), `trace_world`, `system_abcd` |
| `lens_options_dialog.py` | 458 | `apply_real_lens_traced`, inline config table |
| `psf_mtf_dock.py` | 367 | `_PolyStrehlWorker` (QThread), pupil-field synthesis |
| `through_focus_dock.py` | 386 | `ThroughFocusWorker` (QThread), PSF metrics |
| `interferometry_dock.py` | 277 | `simulate_fringes`, OPD-from-wave-optics |
| `phase_retrieval_dock.py` | 266 | `_GSWorker` (QThread), `phase_retrieval_gs` -- *note: Stream C measures 41 LOC of meaningful UI; the 266 includes scaffolding* |
| `caustic_dock.py` | 263 | `CausticWorker` (QThread), `caustic_diagnostic` |
| `rayfan_dock.py` | 224 | `..raytrace` |
| `richards_wolf_dock.py` | 218 | `RichardsWolfWorker` (QThread), `debye_wolf_psf` |
| `shack_hartmann_dock.py` | 191 | `RichardsWolfWorker` (QThread), `shack_hartmann` |
| `jones_pupil_dock.py` | 193 | `compute_jones_pupil`, `plot_jones_pupil` |

### 2.3 Specialised docks (22 files, ~4,234 LOC)

`library_dock.py` (345), `slider_dock.py` (385), `thin_grating_dock.py` (291), `spot_field_dock.py` (253), `distortion_dock.py` (251), `zernike_dock.py` (246), `multiconfig_dock.py` (245), `glass_map_dock.py` (234), `command_palette.py` (233), `diagnostics.py` (222), `field_browser_dock.py` (204), `welcome_dock.py` (197), `footprint_dock.py` (195), `lg_aberration_dock.py` (151), `sensitivity_dock.py` (185), `repl_dock.py` (185), `coherence_dock.py` (162), `snapshots_dock.py` (144), `ghost_dock.py` (141), `materials_dock.py` (45), `__init__.py` (17), and two small helpers.

---

## Part 3 -- Broken-connection scan (Stream B Part 2)

**Result: 0 broken imports across 8 grep classes.**

| Deprecated pattern | Origin | Hits in UI |
|--------------------|--------|------------|
| `from lumenairy.analysis.analysis` | shim removed v5.0.0 | 0 |
| `from lumenairy.ao import ...` | legacy ao module removed v5.0.0; canonical is `lumenairy.analysis.ao` | 0 |
| `from lumenairy.io.hdf5 import ...` | shim removed v5.0.0 | 0 |
| `cosmic_ray_rate=` kwarg | removed v5.0.0 | 0 |
| `install_atexit_restore` direct use | made private v5.2.5; alias kept | 0 |
| `output_grid=` on sub-propagator calls | renamed `output_shape=` v5.2.0 (DeprecationWarning shim alive) | 0 |
| `propagate_huygens_fresnel_freespace` callers assuming bare ndarray | return-type changed to 2-tuple when resampled v5.3.0 | 0 |
| Imports from private modules `_lens_thin` / `_lens_real` / `_lens_traced` | should route through `lumenairy.elements.lenses` shell | 0 |

**Why this is the result.**  The v5.1 large-file splits (~26K LOC across 6 files redistributed into ~35 submodules) intentionally preserved the public-shell re-export contract.  `lumenairy.elements.lenses` continues to re-export `apply_real_lens`, `apply_real_lens_traced`, `apply_thin_lens` directly from the new private files -- the GUI imports the public symbols and never touches the underlying module path.  Same pattern protected the v5.2.0 sub-propagator rename via the DeprecationWarning shim, and the v5.3.0 HF freespace return-type change happens to not be called by any GUI dock (only the `propagate(...)` dispatcher facade is used).

**Verification effort.**  All 45 UI files were grepped exhaustively per pattern using both `from X import Y` and `import X` syntax variants.  False positives (e.g., string keys in config dicts that happen to contain a renamed symbol name) are NOT counted as broken connections -- they are documentation labels, not imports.

This finding inverts the audit's prior hypothesis ("I imagine some of the connections have been broken").  The library's reorganisation discipline + the GUI's reliance on public-shell imports made the breakage class structurally impossible.

---

## Part 4 -- Stale doc-comment markers (Stream B Part 3)

Eleven docks carry version markers (>3 releases stale, library now v5.3.2):

| File | Oldest marker | Apparent staleness | Status |
|------|---------------|---------------------|--------|
| `waveoptics_dock.py` | v3.1.3 | line 99-115 ASM calibration comment cites library v3.2.x | cosmetic |
| `psf_mtf_dock.py` | v4.15 | lines 222 / 236 / 258 sentinel dtype migration | borderline (still accurate) |
| `optimizer_dock.py` | v3.5.0 | line 734 "JAX wave propagator (3.5.0+)" gate | cosmetic |
| `element_table.py` | v3.6.1 | lines 348-1003 stage markers (C.1, C.2) | cosmetic |
| `layout_2d.py` | v3.6.1 hotfix-6 | extensive v3.6.1 -> v3.7.8 fix markers | cosmetic |
| `main_window.py` | v3.5.9 | lines 69-3352 multi-release narrative | cosmetic |
| All other docks | v3.6 - v3.8 | docstring anchors ("Wraps `caustic_diagnostic` (3.5.4)") | cosmetic |

These are NOT broken imports; they are embedded changelog narrative.  No remediation required unless a future cleanup pass elects to strip them.  Future docs-cleanup tickets could collapse the per-release marker style into a single `Author / History` block per file.

---

## Part 5 -- Worker / thread audit (Stream B Part 4)

Ten QThread workers identified across 11 dock files:

| Dock | Worker class | Target library call | Logging handler? | CancellableProgress? |
|------|--------------|---------------------|------------------|----------------------|
| `waveoptics_dock.py` | `WaveOpticsWorker` | `propagate*` (ASM / Fresnel / Fraunhofer) | NO | NO |
| `optimizer_dock.py` | `OptimizeWorker` | `model.run_optimization()` | NO | NO |
| `optimizer_dock.py` | `WaveOptimizeWorker` | propagator + merit loop | NO | NO |
| `optimizer_dock.py` | `GlobalSearchWorker` | global search + merit | NO | NO |
| `tolerance_dock.py` | `ToleranceWorker` | `trace_world()` x N_trials | NO | NO |
| `phase_retrieval_dock.py` | `_GSWorker` | `phase_retrieval_gs()` | NO | NO |
| `psf_mtf_dock.py` | `_PolyStrehlWorker` | pupil-field synth + PSF/Strehl metrics | NO | NO |
| `coherence_dock.py` | `_KoehlerWorker` | `koehler_image()`, `extended_source_image()` | NO | NO |
| `richards_wolf_dock.py` | `RichardsWolfWorker` | `debye_wolf_psf()`, `richards_wolf_focus()` | NO | NO |
| `through_focus_dock.py` | `ThroughFocusWorker` | propagator at varied focal planes | NO | NO |
| `multiconfig_dock.py` | `_MultiConfigWorker` | `trace()` + per-config opt | NO | NO |

**Two structural gaps:**

* **v5.3.2 logging telemetry unconsumed.**  The library's `lumenairy._logging.get_logger(...)` hooks (Newton-iteration residual, scipy iteration progress, MC-trial progress) emit at INFO level with the library's NullHandler.  No worker attaches a handler to forward them.  A 5-line addition per worker -- attach a `logging.Handler` whose `emit` calls `self.progress.emit(record.getMessage())` -- would surface per-iteration diagnostics to the existing status-bar / progress-bar UI for free.

* **CancellableProgress not wired.**  The library exposes a `CancellableProgress` protocol that gates every scipy callback (the v4.13.1 cancellation work).  No GUI worker creates one; no GUI dock surfaces a Stop button.  Long optimisations are run-to-completion only.  Wiring this requires: (a) worker creates `cp = CancellableProgress()`, (b) worker passes `cp` to the library call, (c) dock adds a Stop button whose slot calls `cp.cancel()`, (d) worker's `run()` traps the cancellation exception and emits a clean "cancelled" signal.

Neither gap is a defect of the current code; they are wiring opportunities that the library has been ready for since v4.13.1.

---

## Part 6 -- Library features WITHOUT a GUI surface (Stream C Part 2)

The substantive gap.  Stream A enumerated ~70 user-facing symbols added between v4.13.0 and v5.3.2.  Stream C cross-referenced against existing docks.  The result, by priority:

### 6.1 P1 -- User-blocking gaps

**P1-A. AO module (`lumenairy.analysis.ao`, v5.2.3+).**
Six classes / functions: `DeformableMirror`, `apply_dm()`, `zernike_modal_basis()`, `slope_to_modal()`, `LeakyIntegrator`, `ao_closed_loop()`.  Closed-loop AO is fully usable from the Python API -- but the GUI has no dock to set up DM actuator count, drive a wavefront sensor, or visualise residual convergence per iteration.  Users must prototype in Jupyter.
*Missing dock:* `ao_dock.py` -- DM actuator-count knobs, modal-basis selector, leak / gain / `tol` controls (matches the v5.2.3 `ao_closed_loop` signature including the `gain=0` open-loop fallback path), live residual-vs-iteration plot.

**P1-B. `plot_wavefront()` (analysis.plotting, v4.14.0).**
Rich Zemax-style OPD heatmap with aperture masking, unit conversion (waves / um / mm), colormap control, RMS/PV callout.  Zero docks call it.  Users must export OPD arrays and plot externally.
*Missing dock:* `wavefront_map_dock.py` -- OPD source selector (current system / loaded HDF5 file / live optimiser run), aperture overlay, unit selector, colourmap dropdown, RMS / PV annotation toggles.

**P1-C. Coronagraph workflow (`analysis.coronagraph`).**
`coronagraph_contrast_curve(psf_coro, psf_ref, dx_focal, wavelength, ...)` requires pre-computed PSF pairs.  No interactive Lyot-focal-mask -> Lyot-stop -> apodised-pupil chain builder.
*Missing dock:* `coronagraph_dock.py` -- chain editor (drag-drop the 4 stops), per-stop parameter dialogs (mask radius, stop shape, apodiser profile), live contrast-curve plot.

**P1-D. Optimizer parameter surface (existing `optimizer_dock.py`, 1022 LOC).**
The library `design_optimize()` exposes ~15 parameters: 10+ `method` choices (L-BFGS-B / SLSQP / trust-ncg / DE / basin-hopping / dual_annealing / Nelder-Mead / Powell), `constraints=Sequence[Constraint]` (v4.16.0), `state_file=` checkpoint/resume (v4.16.0), `hess='auto'` Newton-step (v4.16.0), `wave_propagator='real_lens'` (6 backends), `precision='double'`/'single', plus multi-objective Pareto via `design_optimize_multi_objective()` + NSGA-II (v4.16.0 + pymoo optional dep).  The dock hardcodes Nelder-Mead and exposes ~2 of these 15 features (estimated 13% coverage).
*Remediation:* method selector dropdown, constraints editor pane, "Resume from checkpoint" file picker, multi-objective Pareto-mode toggle, hess-callable dropdown, wave-propagator dropdown.

**P1-E. Phase retrieval dock expansion (existing `phase_retrieval_dock.py`, 41 LOC of meaningful UI).**
The library exposes 6 algorithms (`gerchberg_saxton`, `error_reduction`, `hybrid_input_output`, plus JAX twins).  The dock is effectively a stub: algorithm selector hardcoded, no iteration count, no convergence plot, no constraint bounds, no reconstruction preview.  Stream C measures <10% coverage of the library surface.
*Remediation:* expand to ~300-500 LOC: algorithm dropdown (GS / ER / HIO), max-iter spinner, convergence-tolerance spinner, live error-vs-iteration plot, amplitude/phase constraint editors, reconstruction preview tile.

**P1-F. Stop button via CancellableProgress (cross-cutting).**
See Part 5; wire `CancellableProgress` into `optimizer_dock.py` + `tolerance_dock.py` + `phase_retrieval_dock.py` + `multiconfig_dock.py`.  Stop button in each progress bar, slot calls `cp.cancel()`, worker traps and emits cancelled-state.

### 6.2 P2 -- Significant gaps

**P2-A. `lumenairy.algebra` operator algebra (v4.15.1).**
Eight classes (`Operator`, `CompositeOperator`, `FreeSpace`, `ThinLens`, `CylindricalLens`, `Magnify`, `FourierTransform`, `Aperture`, `GaussianAperture`).  Symbolic ABCD matrix algebra; composable via `*` overload; `from_prescription()` classmethod.  No GUI surface -- no operator-chain inspector, no ABCD display, no diagram view.
*Missing dock:* `algebra_dock.py` -- tree view of a CompositeOperator chain, per-node ABCD matrix display, drag-drop reordering, "build chain from current prescription" button.

**P2-B. Logging-telemetry viewer dock (v5.3.2 hooks unconsumed).**
The three long-running paths now emit per-iteration INFO logs.  No dock displays them.
*Missing dock:* `log_viewer_dock.py` -- QPlainTextEdit fed by a `logging.Handler` attached to `lumenairy` root logger, level-selector (DEBUG / INFO / WARN), filter-by-module dropdown.  Pair with the Part 5 worker-handler wiring so the workers' progress signals and the library's iteration logs land in the same widget.

**P2-C. Ghost-path filtering / enumeration (existing `ghost_dock.py`, 141 LOC).**
Library has `ghost_analysis()`, `enumerate_ghost_paths(n_surfaces)`, `non_sequential_stray_light(...)`.  Dock exposes `ghost_analysis()` only.  Path-enumeration matrix, max-bounce filtering, min-transmittance filtering, energetic-weighting ranking all missing.
*Remediation:* add a path-enumeration table view (each row = one ghost path), filter controls in a side panel, sort-by-energy column.

**P2-D. Thin-film coating-stack design.**
Library has `coating_reflectance()`, `quarter_wave_ar()`, `broadband_ar_v_coat()` in `elements.coatings`.  No dock.
*Missing dock:* `coatings_dock.py` -- stack editor (layer table with material + thickness), R(lambda) sweep plot, "fit broadband AR" optimiser button calling `broadband_ar_v_coat()`.

**P2-E. Stokes-parameter polarization visualisation.**
Library has `stokes_parameters(field)` returning S0/S1/S2/S3.  `jones_pupil_dock.py` covers Jones representation; no Stokes view.
*Remediation:* either extend `jones_pupil_dock.py` with a "Stokes view" tab or add `stokes_dock.py` companion.  Four S0..S3 heatmaps + degree-of-polarisation derived map.

### 6.3 P3 -- Polish / lower priority

**P3-A. Coherence dock expansion (existing `coherence_dock.py`, 162 LOC).**
Currently focused on Schell-model source creation.  Library has `koehler_image()`, `extended_source_image()`, `mutual_coherence()` analyses that aren't wired.  Add a tab for each.

**P3-B. Zernike dock controls (existing `zernike_dock.py`, 246 LOC).**
Library supports `normalization='OSC'` / `'Fringe'` / `'Standard'`; dock hardcodes OSC.  No edge-weighting mask UI.  Add a normalisation selector + weighting-mask file-picker.

**P3-C. Chebyshev surface fitter (`_math.chebyshev`, v5.2.0 extraction).**
Six library sites use the shared Chebyshev helpers but no UI offers a "fit measured freeform data to Chebyshev polynomials" tool.  Specialised metrology workflow; build on demand.

---

## Part 7 -- GUI surfaces under-exposing their library (Stream C Part 3)

Seven docks measured for parameter-coverage ratio.  P1 and P2 entries already appear in Part 6; tabulated here for compactness:

| Dock | LOC | Library params / modes | Dock exposure | Coverage | Priority |
|------|-----|-------------------------|---------------|----------|----------|
| `optimizer_dock.py` | 1022 | ~15 (method, constraints, checkpoint, hess, multi-obj, wave-propagator, precision...) | ~2 | 13% | P1 |
| `phase_retrieval_dock.py` | 41 (meaningful) | 8+ (algorithm, iters, convergence, constraints, JAX variants) | <1 | <10% | P1 |
| `ghost_dock.py` | 141 | 3 functions + filtering / ranking knobs | 1 function | 17% | P2 |
| `tolerance_dock.py` | 524 | ~8 (distribution, error_callback, merit_fn, correlation, wave-vs-ray) | ~4 | 50% | P2 |
| `psf_mtf_dock.py` | 367 | 6+ (polychromatic, MTF method, metric extraction, mode-removal) | ~4 | 40% | P2 |
| `coherence_dock.py` | 162 | 3 analysis fns + Schell params | ~0.5 | 30% | P3 |
| `zernike_dock.py` | 246 | 6+ (normalisation, poly order, weighting mask, scale) | ~2 | 35% | P3 |

**Worst offender:** `optimizer_dock.py` at 13% parameter coverage of the central library feature.  Closing this gap unblocks the bulk of the v4.16.0 optimisation framework that nobody touches in the GUI today (constraints, checkpoint/resume, Newton, multi-objective Pareto).

---

## Part 8 -- ROADMAP corrections

Current ROADMAP.md "Designer GUI (separate v3.8+ stream)" section lists 6 unscoped items.  After this audit, the correct picture is:

| ROADMAP claim | Actual status | Correction |
|---------------|---------------|------------|
| 1. Polarization plotting docks (none surface Jones-pupil + Stokes) | `jones_pupil_dock.py` shipped (193 LOC, complete); Stokes still missing | Update to "Stokes-parameter docks (Jones-pupil shipped at v3.6+)" |
| 2. Coronagraph workflow dock | confirmed missing | keep |
| 3. Tolerancing dock | `tolerance_dock.py` shipped at 524 LOC -- ROADMAP CLAIM IS WRONG | remove from missing-list; move to shipped + add P2 expansion item per Part 7 |
| 4. Multi-config / zoom dock | `multiconfig_dock.py` shipped at 245 LOC -- ROADMAP CLAIM IS WRONG | remove from missing-list; move to shipped |
| 5. Wavefront-map plot integration | confirmed missing | keep |
| 6. CancellableProgress UI button | library hook ready since v4.13.1, GUI side missing | keep |

**Additional missing items not currently in ROADMAP that this audit surfaces:**

| New ROADMAP item | Priority | Source |
|------------------|----------|--------|
| AO closed-loop dock (`ao_closed_loop` v5.2.3 unsurfaced) | P1 | Part 6.1 P1-A |
| Optimizer dock parameter surface (v4.16.0 constraints / checkpoint / Newton / multi-obj) | P1 | Part 6.1 P1-D + Part 7 |
| Phase-retrieval dock expansion (current 41 LOC stub -> ~300-500 LOC) | P1 | Part 6.1 P1-E + Part 7 |
| Logging-telemetry viewer dock (v5.3.2 hooks unconsumed) | P2 | Part 5 + Part 6.2 P2-B |
| Algebra operator-chain inspector (`lumenairy.algebra` v4.15.1 unsurfaced) | P2 | Part 6.2 P2-A |
| Ghost-path enumeration / filtering UI | P2 | Part 6.2 P2-C |
| Coatings-stack design dock | P2 | Part 6.2 P2-D |
| Stokes-parameter visualisation | P2 | Part 6.2 P2-E |
| Coherence dock expansion (analysis fns) | P3 | Part 6.3 P3-A |
| Zernike normalisation / weighting-mask controls | P3 | Part 6.3 P3-B |

---

## Part 9 -- Recommended remediation sequencing

**Tier 1 -- v3.8.0 (P1, user-blocking).** 6 items.  Aggregate ~3000-5000 GUI LOC.

1. **Stop-button wiring** (P1-F).  Lowest-LOC, highest-cross-cutting value; do first as a structural enabler for all subsequent long-running docks.  ~50-100 LOC per dock for 4 docks.
2. **Optimizer dock parameter surface** (P1-D).  Highest under-exposure on a central feature.  ~500-800 LOC of new widgets.
3. **Wavefront-map dock** (P1-B).  Single new dock; wraps a single library function richly.  ~400-600 LOC.
4. **Phase-retrieval dock expansion** (P1-E).  Convert stub to real dock.  ~500-800 LOC.
5. **AO closed-loop dock** (P1-A).  New dock; ~600-900 LOC.
6. **Coronagraph workflow dock** (P1-C).  New dock with the 4-stop chain builder.  ~800-1200 LOC.

**Tier 2 -- v3.9.0 (P2, significant).** 5 items.  Aggregate ~2000-3000 GUI LOC.

7. Logging-telemetry viewer dock (P2-B).  ~200-300 LOC + per-worker handler wiring.
8. Algebra operator-chain inspector (P2-A).  ~600-900 LOC.
9. Ghost-path enumeration / filtering UI (P2-C).  ~300-500 LOC expansion.
10. Coatings-stack design dock (P2-D).  ~500-800 LOC.
11. Stokes-parameter visualisation tab (P2-E).  ~200-400 LOC.

**Tier 3 -- v3.10.0+ (P3, polish).** 3 items.

12. Coherence dock expansion (P3-A).
13. Zernike normalisation / weighting-mask controls (P3-B).
14. Chebyshev freeform-fit tool (P3-C).

**Cross-cutting hygiene (any release):**
- Strip stale v3.x-era comments from `main_window.py` / `layout_2d.py` / `element_table.py` (Part 4).  Cosmetic, low priority.
- Wire `logging.Handler` into each QThread worker so library INFO logs reach the UI (Part 5).  ~5-line addition per worker x 10 workers = ~50 LOC total.

---

## Part 10 -- Cross-audit observations

* **Library reorganisation hygiene worked.**  The v5.0 / v5.1 / v5.2 / v5.3 reorganisation discipline (shell-shim pattern + DeprecationWarning shims + private-module underscore prefix) was strong enough that zero GUI imports broke across four major releases.  This is a real signal -- the library's structural-walker family (V1-V18) is not just CI noise; the V9 `__all__` symmetry pin + the V13 shell-vs-canonical pin + the V14 PEP-562 forwarding pin together build the contract that keeps GUI imports valid.  Future reorganisations can lean on the same pattern.

* **The Designer is behind by 2-3 minor releases of library work.**  The latest dock implementations cite v3.7.10 features; the latest substantive library features (v4.16.0 optimisation framework, v5.2.3 AO closed-loop, v5.3.2 logging telemetry) have no dock surfaces.  This is normal -- the library has shipped 15 minor releases in the v4.13.0 -> v5.3.2 window; the Designer's last named version bump (3.7.10) is approximately a year stale by library cadence.  The 14-item Tier 1+2+3 sequencing above closes the gap.

* **The ROADMAP's "Designer GUI (separate version stream)" section is itself stale.**  It claims two docks are missing that are full implementations.  Recommendation: regenerate that section from this audit.

* **No new audit-class meta-walker required.**  The audit found gaps but no novel sibling-gap pattern.  The "GUI under-exposes library function" class is already implicitly tracked by the lack-of-a-test-pin -- a future walker that diffs library `__all__` symbols against GUI dock imports could automate Part 6's detection, but adding it now would add CI weight before the gap is materially closed.  Build the walker AFTER Tier 1 + Tier 2 ship.

* **The "feature complete for v5.x" claim from v5.3.2 ROADMAP is correct for the LIBRARY but does not cover the GUI.**  The ROADMAP's "v5.x ROADMAP code-work is fully closed" is honest about library scope; this audit makes the GUI side of the gap explicit and assignable.

---

## Appendix A -- Library feature inventory (Stream A condensed)

For traceability.  Full inventory has ~70 entries; grouped by category and tagged by introduction version.

**Analysis (14 symbols):** `encircled_energy_curve` v4.14.0, `encircled_energy_radius` v4.14.0, `mtf_cutoff` v4.14.0, `beam_diameter` v4.14.0, `depth_of_focus` v4.14.0, `rayleigh_resolution` v4.15.0, `sparrow_resolution` v4.15.0, `fwhm_resolution` v4.15.0, `astigmatism_mag_angle` v4.15.0, `plot_wavefront` v4.14.0, `plot_opd_fan` v4.15.4, `plot_opd_summary` v4.15.4, `ee_polychromatic` v4.15.0, `strehl_vector` v4.15.0, `coupling_efficiency_vector` v4.15.0.

**Sources / partial coherence (6):** `create_gaussian_schell_source` v4.15.0, `create_schell_model_source` v4.15.0, `create_annular_incoherent_source` v4.15.0, `PartialCoherenceMCF` v4.15.1, `MCF` alias v5.2.0, `propagate_ensemble` v4.16.1.

**Elements / surfaces (4):** `surface_sag_q_bfs` v4.15.0, `make_off_axis_parabola` v4.15.0 (chief-ray fix v4.15.1), `apply_real_lens_maslov` v4.16+, `apply_real_lens_maslov_jax` v4.16+.

**Optimisation (7):** `Constraint` v4.16.0, `design_optimize_multi_objective` v4.16.0, `ParetoResult` v4.16.0, `PYMOO_AVAILABLE` v4.16.0, `WAVE_PROPAGATOR_REGISTRY` v4.16.0, `register_wave_propagator` v4.16.0, `unregister_wave_propagator` v4.16.0.

**Algebra (CLUSTER_B, 9):** `Operator`, `CompositeOperator`, `FreeSpace`, `ThinLens`, `CylindricalLens`, `Magnify`, `FourierTransform`, `Aperture`, `GaussianAperture` -- all v4.15.1.

**Adaptive optics (1 high-level):** `ao_closed_loop` v5.2.3 (with v5.2.5 `leak` / `tol` / open-loop / skip-update extensions).

**Raytrace bridge (1):** `rays_from_field` v4.15.1.

**Glass / materials:** `GLASS_VALIDITY` v4.16.0; formula-3 evaluator + 24 new glass coefficients v5.2.0.

**Multi-config (5):** `Configuration`, `create_zoom_configs`, `multi_config_merit`, `afocal_angular_magnification`, `beam_expander_prescription`, `keplerian_telescope` -- public surfaces consolidated in v5.1+.

**System (1):** `evaluate` v4.15.0 -- one-call prescription + Source -> PropagationResult entry.

**Infrastructure (logging + cache + defaults):** `register_cache_clearer` / `list_registered_cache_clearers` v4.16.0; `DEFAULT_REAL_DTYPE` / `DEFAULT_WAVE_PROPAGATOR` / `DEFAULT_DY` top-level re-exports v4.16.2; v5.3.2 per-iteration logging hooks on three long-running paths.

**Tolerancing (1):** `monte_carlo_tolerancing_linearized` v4.13.1+.

Total user-facing: ~70 symbols.  GUI-surfaced today: a substantial fraction in `model.py` and the existing 23 docks; the gap-list in Parts 6-7 captures the high-relevance subset that needs explicit dock surfaces.

---

**End of audit.**

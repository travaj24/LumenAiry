# WP-A17 SWEEP-3 -- version-history relocation, library-wide (my partition)

Scope of this sweep: **comments and docstrings only, zero executable change**, over
the root modules `lumenairy/*.py` (except `__init__.py` and `memory.py`), and
everything under `lumenairy/io/`, `lumenairy/optimize/`, `lumenairy/raytrace/`,
`lumenairy/ui/` (except `lens_options_dialog.py`), `lumenairy/backend/`
(except `backend/__init__.py`), `lumenairy/_math/` and `lumenairy/algebra/`.

Finding: **P2-4** (`TESTS-ARCH.md:394`) / consolidated report **sec. 14 V6** and
**sec. 15.7**.  Method and document format follow WP-A17 part 1
(`fixes/WP-A17_REPORT.md`); the checker is
`tests/unit/test_audit2609_a17_history_relocation.py`, unchanged by me.

---

## 1. Summary

| | |
|---|---|
| modules in the partition | **111** (fingerprinted before any edit) |
| modules changed | **94** |
| modules deliberately left alone | **17** (sec. 6) |
| lines | **68 900 -> 68 577** (-323) |
| LOOSE history lines (the audit's own classifier) | **9 350 (13.6 %) -> 5 799 (8.5 %)** |
| STRICT history lines (`vN.N (`, `pre-fix`, `used to`, `formerly`, `previously`, `the old`, `was wrong`, ...) | **843 -> 158 (-81 %)** |
| history documents written | **45** (`docs/history/<dotted.module.path>.md`) |
| blocks recorded verbatim | **444** (7 270 document lines) |
| release TAGS stripped in place (no narrative attached) | **287** across **69** modules -- full before/after list in sec. 8 |
| contradicting comments corrected | **3** (sec. 4) |
| retired prose assertions | **none** |

**Both fingerprints are byte-for-byte identical to the pre-relocation file on
all 111 modules.**  Not "equivalent" -- identical, verified after every apply
and once more at the end:

```
111/111 identical
```

`ruff check` clean across the whole partition; `python -c "import lumenairy"`
OK; all 233 modules under `lumenairy/` compile.

### 1.1 Method

Per module, biggest first:

1. record both fingerprints (`ast_fingerprint` / `token_fingerprint`, the
   checker's own helpers) from the UNTOUCHED file;
2. move every history block verbatim into
   `docs/history/<dotted.module.path>.md` under the source line it came from,
   with a line-ordered TOC and a *Left in the source:* note per block;
3. leave a pointer only where the old rationale still explains present
   behaviour;
4. re-compute both fingerprints and REFUSE TO WRITE unless they are identical
   -- the apply tool enforces this, so a mistake cannot reach disk;
5. `ruff check` the module; re-import the package.

The dotted document name (`docs/history/lumenairy.io.storage.md`) is the
convention the orchestrator generalised the checker for.  It works: the name
assertion now derives the dotted path from the header's repo-relative
`module:` field, and every one of my 45 documents passes it.

### 1.2 The three patterns that carried the history here

**(a) A running log appended to, never updated.**  `_deprecation.py` is the
pure case and the single largest relocation in the sweep: a five-entry
horizon-slip log on `NEXT_REMOVAL_VERSION` (v5.32.0, v5.36.1, v5.40.0,
v5.43.0, v5.45.0 -- each recording that the horizon came due with both
registries empty and was advanced by one line), a W5 execution log listing
every shim a wave removed, and two tombstones describing retired registry
entries.  688 -> 619 lines, and `NEXT_REMOVAL_VERSION` is still `'5.48'`.

**(b) The same fix annotated at every site it touched.**  The grating
`1 / n2` factor appears in four modules; the `_ZERO_APERTURE_MASK`
zero-aperture semantics in four; the intensity-threshold `>=` convention at
six sites in one module; the writer version stamp at nine sites in
`io/storage.py`.  In every case the RULE and its measured consequence are
live and stayed in the source, re-stated in the present tense; what moved is
the per-site record of which release wrote which half.

**(c) A boilerplate release tag on an otherwise-live why-comment.**  287 of
these, 111 of them the same three lines repeated across the UI dock family
(`v5.4.3 (audit GUI-resize)` / `v5.4.4 (audit GUI-resize round 2)`).  These
are one-line labels with no narrative attached, so relocating them into 40
new documents plus 40 new source pointers would have added more noise than it
removed: they were stripped in place, and the complete verbatim before/after
list is sec. 8 of this report.  The rule is stated once here and applied
uniformly; a line whose remainder carried any narrative marker (`pre-fix`,
`used to`, `formerly`, `previously`, `the old`, `no longer`, `historic...`)
was excluded from the mechanical pass and relocated by hand instead.

### 1.3 What deliberately did NOT move

* **Measured derivations of live values.**  `docs/TESTING_STANDARDS.md` S5
  requires a numeric bar to carry its oracle.  Kept in full: the
  `_GLASS_VALUE_CACHE_SIZE` 65536 sizing argument, the wrapper-merit grid
  cache's 7.6 GB-at-N=2048 retention figure, the Zemax EVENASPH power
  derivation with its +2.8 mm poc1-19/20 defocus, the seidel sign conventions
  with their exact-trace oracles (<= 1.7e-11, -0.9975...-0.9998), the grating
  `1 / n2` ratio 1.503583 = n(N-BK7), `deep_nbytes`'s 64x over-budget and
  8000 B double-count, the OAP `h = 2 f tan(alpha)` derivation, the
  compression measurements (57.9x write, 7.1x read, 5.6 % of space), the
  Chebyshev / LG normalisation figures (4.79e+14 and 3.2e-03).
* **Live migration statements.**  `.. versionchanged:: 5.30` on
  `design_optimize` / `MatchIdealSystem` (five kwargs now raise `TypeError`,
  with the migration recipe and the grep-verification that made the removal
  safe); `io/storage.py`'s `(v5.46 default)` / "pass `compression='gzip'` to
  restore the pre-v5.46 behaviour"; `optimize/context.py`'s one-cycle
  `DeprecationWarning` on the removed `Constraint` auto-probe; `_context.py`'s
  `install_atexit_restore` back-compat alias.  These tell a user what to do
  now.
* **`io/prescriptions_code_v.py` entirely** (16 strict lines left).  Its
  "pre-v5.46" references are not narrative: a pre-v5.46 file really is on disk
  carrying `DIM M` with SI-metre numbers, the reader really does detect it
  from this writer's banner, and `_CV_LEGACY_TO_METERS` is the live table that
  reads it.  That is a file-format migration contract.
* **`optimize/core.py` and `raytrace/core.py` entirely.**  Both are re-export
  shells whose comment blocks are a SOURCE-GREP CONTRACT: five separate tests
  (`test_v4_15_agent_f`, `test_v4_16_1_fix_lines_present`,
  `test_dual_annealing_callback_signature_polls_is_cancelled`,
  `test_changelog_v4_15_2_sentinel_line_citations_refreshed`,
  `test_v4_15_2_agent_a::test_changelog_optimize_core_line_citation_refreshed`)
  grep those files for literal strings the v5.1.0 split moved into submodules.
  Editing the prose would retire real assertions for no gain.
* **`optimizer_dock.py`'s "Nelder-Mead because `model.run_optimization()`
  hardcoded it" statements** (15 strict lines left).  They are the reason a
  LIVE default is what it is -- the first thing a reader changing that default
  needs.

---

## 2. Per-module numbers

`before` is `git show HEAD:<module>` (the pre-relocation file); `after` is the
delivered working tree.  LOOSE is the audit's own classifier (a comment block
counts if it mentions a version, the word "audit" or a `20xx-xx-xx` date; a
docstring if it carries >= 2 such markers).  STRICT is the per-LINE count of
the shapes the finding actually names.  `blocks` is the number of verbatim
blocks in that module's history document (`--` = tag-only strips, sec. 8).

| module | lines b -> a | loose history b -> a | strict lines b -> a | blocks |
|---|---|---|---|---|
| `lumenairy/_deprecation.py` | 688 -> 619 | 234 (34.0 %) -> 137 (22.1 %) | 19 -> 5 | 14 |
| `lumenairy/glass.py` | 2160 -> 2092 | 562 (26.0 %) -> 305 (14.6 %) | 45 -> 3 | 37 |
| `lumenairy/io/storage.py` | 2159 -> 2109 | 465 (21.5 %) -> 134 (6.4 %) | 38 -> 10 | 33 |
| `lumenairy/optimize/driver.py` | 1674 -> 1645 | 475 (28.4 %) -> 294 (17.9 %) | 37 -> 1 | 37 |
| `lumenairy/optimize/wrapper_merits.py` | 1048 -> 1021 | 433 (41.3 %) -> 163 (16.0 %) | 42 -> 1 | 31 |
| `lumenairy/io/prescriptions_zemax.py` | 2830 -> 2808 | 362 (12.8 %) -> 125 (4.5 %) | 45 -> 3 | 43 |
| `lumenairy/optimize/context.py` | 605 -> 587 | 214 (35.4 %) -> 39 (6.6 %) | 20 -> 1 | 16 |
| `lumenairy/raytrace/from_field.py` | 943 -> 934 | 407 (43.2 %) -> 192 (20.6 %) | 13 -> 0 | 10 |
| `lumenairy/io/prescriptions_builders.py` | 660 -> 653 | 62 (9.4 %) -> 60 (9.2 %) | 6 -> 0 | 5 |
| `lumenairy/raytrace/jax_trace.py` | 1821 -> 1814 | 403 (22.1 %) -> 227 (12.5 %) | 29 -> 1 | 24 |
| `lumenairy/_math/chebyshev.py` | 456 -> 450 | 40 (8.8 %) -> 0 (0.0 %) | 15 -> 3 | 9 |
| `lumenairy/raytrace/seidel.py` | 1913 -> 1907 | 290 (15.2 %) -> 273 (14.3 %) | 24 -> 8 | 17 |
| `lumenairy/_validation.py` | 274 -> 270 | 153 (55.8 %) -> 63 (23.3 %) | 9 -> 0 | 4 |
| `lumenairy/backend/fft.py` | 253 -> 249 | 51 (20.2 %) -> 43 (17.3 %) | 8 -> 0 | 4 |
| `lumenairy/optimize/merit_terms.py` | 2002 -> 1999 | 757 (37.8 %) -> 641 (32.1 %) | 18 -> 1 | 18 |
| `lumenairy/ui/model.py` | 3522 -> 3519 | 253 (7.2 %) -> 191 (5.4 %) | 26 -> 7 | 9 |
| `lumenairy/optimize/multi_objective.py` | 450 -> 448 | 14 (3.1 %) -> 15 (3.3 %) | 4 -> 2 | 3 |
| `lumenairy/raytrace/differential.py` | 1078 -> 1076 | 81 (7.5 %) -> 81 (7.5 %) | 4 -> 0 | 3 |
| `lumenairy/raytrace/trace.py` | 1902 -> 1900 | 240 (12.6 %) -> 197 (10.4 %) | 17 -> 3 | 9 |
| `lumenairy/_context.py` | 365 -> 364 | 52 (14.2 %) -> 31 (8.5 %) | 11 -> 2 | 4 |
| `lumenairy/io/prescriptions.py` | 106 -> 105 | 5 (4.7 %) -> 0 (0.0 %) | 3 -> 0 | -- |
| `lumenairy/optimize/jax_merits.py` | 806 -> 805 | 185 (23.0 %) -> 172 (21.4 %) | 7 -> 0 | 7 |
| `lumenairy/optimize/multiconfig.py` | 483 -> 482 | 29 (6.0 %) -> 20 (4.1 %) | 11 -> 4 | 6 |
| `lumenairy/user_library.py` | 1271 -> 1270 | 101 (7.9 %) -> 15 (1.2 %) | 15 -> 1 | 10 |
| `lumenairy/_cache_registry.py` | 292 -> 292 | 112 (38.4 %) -> 112 (38.4 %) | 5 -> 3 | 2 |
| `lumenairy/_knobs.py` | 357 -> 357 | 71 (19.9 %) -> 71 (19.9 %) | 1 -> 0 | -- |
| `lumenairy/_logging.py` | 51 -> 51 | 26 (51.0 %) -> 4 (7.8 %) | 2 -> 1 | -- |
| `lumenairy/algebra/__init__.py` | 133 -> 133 | 28 (21.1 %) -> 0 (0.0 %) | 2 -> 1 | -- |
| `lumenairy/algebra/apertures.py` | 210 -> 210 | 5 (2.4 %) -> 5 (2.4 %) | 2 -> 1 | -- |
| `lumenairy/algebra/base.py` | 566 -> 566 | 41 (7.2 %) -> 35 (6.2 %) | 1 -> 0 | -- |
| `lumenairy/algebra/from_prescription.py` | 230 -> 230 | 20 (8.7 %) -> 2 (0.9 %) | 3 -> 2 | 1 |
| `lumenairy/backend/array.py` | 259 -> 259 | 9 (3.5 %) -> 6 (2.3 %) | 2 -> 1 | -- |
| `lumenairy/backend/random.py` | 254 -> 254 | 38 (15.0 %) -> 12 (4.7 %) | 11 -> 2 | -- |
| `lumenairy/backend/scipy.py` | 212 -> 212 | 36 (17.0 %) -> 32 (15.1 %) | 3 -> 2 | -- |
| `lumenairy/io/codegen.py` | 1221 -> 1221 | 79 (6.5 %) -> 66 (5.4 %) | 15 -> 6 | 2 |
| `lumenairy/io/prescriptions_code_v.py` | 923 -> 923 | 85 (9.2 %) -> 74 (8.0 %) | 19 -> 16 | -- |
| `lumenairy/optimize/__init__.py` | 136 -> 136 | 10 (7.4 %) -> 1 (0.7 %) | 2 -> 0 | -- |
| `lumenairy/optimize/_merit_jit.py` | 273 -> 273 | 26 (9.5 %) -> 23 (8.4 %) | 8 -> 0 | 7 |
| `lumenairy/progress.py` | 246 -> 246 | 79 (32.1 %) -> 76 (30.9 %) | 1 -> 0 | -- |
| `lumenairy/raytrace/__init__.py` | 204 -> 204 | 12 (5.9 %) -> 7 (3.4 %) | 3 -> 1 | -- |
| `lumenairy/raytrace/intersection.py` | 843 -> 843 | 186 (22.1 %) -> 181 (21.5 %) | 15 -> 2 | 11 |
| `lumenairy/raytrace/surface.py` | 751 -> 751 | 114 (15.2 %) -> 96 (12.8 %) | 7 -> 3 | 2 |
| `lumenairy/ui/__init__.py` | 22 -> 22 | 22 (100.0 %) -> 0 (0.0 %) | 3 -> 2 | -- |
| `lumenairy/ui/algebra_dock.py` | 959 -> 959 | 30 (3.1 %) -> 21 (2.2 %) | 3 -> 1 | -- |
| `lumenairy/ui/ao_dock.py` | 955 -> 955 | 57 (6.0 %) -> 41 (4.3 %) | 6 -> 2 | -- |
| `lumenairy/ui/caustic_dock.py` | 291 -> 291 | 9 (3.1 %) -> 0 (0.0 %) | 2 -> 0 | -- |
| `lumenairy/ui/chebyshev_fit_dock.py` | 574 -> 574 | 32 (5.6 %) -> 23 (4.0 %) | 2 -> 0 | -- |
| `lumenairy/ui/coatings_dock.py` | 782 -> 782 | 72 (9.2 %) -> 63 (8.1 %) | 3 -> 1 | -- |
| `lumenairy/ui/coherence_dock.py` | 872 -> 872 | 85 (9.7 %) -> 74 (8.5 %) | 9 -> 3 | 2 |
| `lumenairy/ui/coronagraph_dock.py` | 851 -> 851 | 67 (7.9 %) -> 50 (5.9 %) | 5 -> 1 | -- |
| `lumenairy/ui/distortion_dock.py` | 271 -> 271 | 22 (8.1 %) -> 13 (4.8 %) | 2 -> 0 | -- |
| `lumenairy/ui/element_table.py` | 1376 -> 1376 | 51 (3.7 %) -> 43 (3.1 %) | 3 -> 2 | -- |
| `lumenairy/ui/field_browser_dock.py` | 225 -> 225 | 9 (4.0 %) -> 0 (0.0 %) | 2 -> 0 | -- |
| `lumenairy/ui/footprint_dock.py` | 215 -> 215 | 15 (7.0 %) -> 6 (2.8 %) | 2 -> 0 | -- |
| `lumenairy/ui/ghost_dock.py` | 728 -> 728 | 38 (5.2 %) -> 28 (3.8 %) | 4 -> 1 | -- |
| `lumenairy/ui/glass_map_dock.py` | 255 -> 255 | 10 (3.9 %) -> 1 (0.4 %) | 2 -> 0 | -- |
| `lumenairy/ui/interferometry_dock.py` | 292 -> 292 | 9 (3.1 %) -> 0 (0.0 %) | 2 -> 0 | -- |
| `lumenairy/ui/jones_pupil_dock.py` | 379 -> 379 | 41 (10.8 %) -> 6 (1.6 %) | 6 -> 0 | -- |
| `lumenairy/ui/lg_aberration_dock.py` | 207 -> 207 | 31 (15.0 %) -> 15 (7.2 %) | 5 -> 3 | -- |
| `lumenairy/ui/library_dock.py` | 363 -> 363 | 8 (2.2 %) -> 0 (0.0 %) | 1 -> 0 | -- |
| `lumenairy/ui/log_viewer_dock.py` | 410 -> 410 | 10 (2.4 %) -> 2 (0.5 %) | 2 -> 1 | -- |
| `lumenairy/ui/materials_dock.py` | 63 -> 63 | 8 (12.7 %) -> 0 (0.0 %) | 2 -> 1 | -- |
| `lumenairy/ui/multiconfig_dock.py` | 313 -> 313 | 31 (9.9 %) -> 0 (0.0 %) | 6 -> 0 | -- |
| `lumenairy/ui/phase_retrieval_dock.py` | 923 -> 923 | 30 (3.3 %) -> 0 (0.0 %) | 10 -> 1 | -- |
| `lumenairy/ui/psf_mtf_dock.py` | 469 -> 469 | 43 (9.2 %) -> 34 (7.2 %) | 6 -> 2 | -- |
| `lumenairy/ui/rayfan_dock.py` | 244 -> 244 | 19 (7.8 %) -> 10 (4.1 %) | 2 -> 0 | -- |
| `lumenairy/ui/repl_dock.py` | 203 -> 203 | 10 (4.9 %) -> 2 (1.0 %) | 1 -> 0 | -- |
| `lumenairy/ui/richards_wolf_dock.py` | 281 -> 281 | 18 (6.4 %) -> 0 (0.0 %) | 5 -> 1 | -- |
| `lumenairy/ui/sensitivity_dock.py` | 206 -> 206 | 12 (5.8 %) -> 3 (1.5 %) | 2 -> 0 | -- |
| `lumenairy/ui/shack_hartmann_dock.py` | 234 -> 234 | 20 (8.5 %) -> 11 (4.7 %) | 3 -> 1 | -- |
| `lumenairy/ui/slider_dock.py` | 433 -> 433 | 32 (7.4 %) -> 2 (0.5 %) | 3 -> 0 | -- |
| `lumenairy/ui/snapshots_dock.py` | 162 -> 162 | 8 (4.9 %) -> 0 (0.0 %) | 1 -> 0 | -- |
| `lumenairy/ui/spot_field_dock.py` | 294 -> 294 | 22 (7.5 %) -> 13 (4.4 %) | 2 -> 0 | -- |
| `lumenairy/ui/thin_grating_dock.py` | 311 -> 311 | 9 (2.9 %) -> 0 (0.0 %) | 2 -> 0 | -- |
| `lumenairy/ui/through_focus_dock.py` | 434 -> 434 | 27 (6.2 %) -> 18 (4.1 %) | 2 -> 0 | -- |
| `lumenairy/ui/tolerance_dock.py` | 599 -> 599 | 50 (8.3 %) -> 16 (2.7 %) | 9 -> 0 | -- |
| `lumenairy/ui/wavefront_map_dock.py` | 691 -> 691 | 24 (3.5 %) -> 9 (1.3 %) | 5 -> 2 | -- |
| `lumenairy/ui/welcome_dock.py` | 215 -> 215 | 16 (7.4 %) -> 8 (3.7 %) | 1 -> 0 | -- |
| `lumenairy/ui/zernike_dock.py` | 427 -> 427 | 20 (4.7 %) -> 11 (2.6 %) | 4 -> 2 | -- |
| `lumenairy/algebra/primitives.py` | 836 -> 837 | 257 (30.7 %) -> 218 (26.0 %) | 14 -> 3 | 4 |
| `lumenairy/cache.py` | 676 -> 677 | 33 (4.9 %) -> 4 (0.6 %) | 1 -> 0 | 1 |
| `lumenairy/io/prescriptions_quadoa.py` | 440 -> 441 | 63 (14.3 %) -> 10 (2.3 %) | 8 -> 0 | 4 |
| `lumenairy/optimize/parameterizations.py` | 479 -> 480 | 45 (9.4 %) -> 0 (0.0 %) | 11 -> 0 | 11 |
| `lumenairy/raytrace/bundles.py` | 227 -> 228 | 5 (2.2 %) -> 5 (2.2 %) | 3 -> 0 | 2 |
| `lumenairy/raytrace/layout.py` | 190 -> 191 | 10 (5.3 %) -> 10 (5.2 %) | 2 -> 0 | 2 |
| `lumenairy/raytrace/paraxial.py` | 306 -> 307 | 46 (15.0 %) -> 0 (0.0 %) | 4 -> 0 | 3 |
| `lumenairy/raytrace/world_trace.py` | 250 -> 251 | 24 (9.6 %) -> 24 (9.6 %) | 3 -> 0 | 3 |
| `lumenairy/ui/main_window.py` | 3645 -> 3646 | 189 (5.2 %) -> 169 (4.6 %) | 8 -> 2 | 2 |
| `lumenairy/ui/optimizer_dock.py` | 2030 -> 2031 | 234 (11.5 %) -> 82 (4.0 %) | 46 -> 15 | 3 |
| `lumenairy/ui/waveoptics_dock.py` | 3324 -> 3325 | 306 (9.2 %) -> 263 (7.9 %) | 18 -> 4 | 12 |
| `lumenairy/io/prescriptions_transforms.py` | 742 -> 744 | 54 (7.3 %) -> 46 (6.2 %) | 13 -> 7 | 6 |
| `lumenairy/raytrace/seidel_analysis.py` | 405 -> 407 | 50 (12.3 %) -> 50 (12.3 %) | 2 -> 1 | 1 |
| `lumenairy/ui/layout_2d.py` | 1118 -> 1120 | 109 (9.7 %) -> 109 (9.7 %) | 5 -> 1 | 4 |
| `lumenairy/raytrace/ray_fan.py` | 1073 -> 1076 | 63 (5.9 %) -> 65 (6.0 %) | 7 -> 1 | 6 |
| **total (94 modules)** | **68900 -> 68577** | **9350 (13.6 %) -> 5799 (8.5 %)** | **843 -> 158** | **444** |

---

## 3. Files touched

**Modified (comments and docstrings only -- AST and token stream identical on
every one):** the 94 modules in the table above.

**New:** 45 documents under `docs/history/`, one per module, listed in the
`blocks` column of the table above.

**Not modified:** the 17 modules of sec. 6, plus every test file -- see sec. 5b
and "Retired prose assertions: none" below.

**Retired prose assertions: none.**  Every prose pin found in the 60-file
source-reading sweep is a pin on a statement that is still true of the current
behaviour, so keeping the statement was both cheaper and more honest than
retiring the test.  Two pins were actively protected:

* `tests/unit/test_v4_16_3_agent_d.py::test_glass_dispatch_order_comment_matches_code`
  requires the literal `SELLMEIER -> POLYNOMIAL` in `glass.py`; the condensed
  comment keeps it.
* `tests/unit/test_v4_15_agent_f.py::test_changelog_cites_actual_current_lines`
  re-derives `merit_terms.py`'s `np.isfinite(ap) and ap > 0` line by anchor
  string and requires a CHANGELOG citation within +/- 5 of it.  Every edit
  ABOVE that anchor in `merit_terms.py` is line-count neutral ON PURPOSE, so
  the anchor has not moved.

---

## 4. Contradicting comments corrected (audit sec. 15.7, "worse than none")

Three comments in this partition stated something the code beneath them does
not do.  All three are comment-only edits and are covered by the identity gate.

1. **`lumenairy/glass.py:1906`** (`get_glass_index`, refractiveindex-unavailable
   fallback).  The comment read *"formula-3 polynomial fallback ... Empty at
   v4.16.2 ship; populating the 24 catalogue entries is staged for v5.2.1"*.
   `POLYNOMIAL_COEFFICIENTS` has held all 24 entries since v5.2.3 -- the
   table's own preamble says so, and `_POLYNOMIAL_STUB_NAMES` is an empty
   frozenset -- so the arm it annotates covers every registered formula-3 glass
   on a minimal install.  Corrected to say that.
2. **`lumenairy/glass.py:1598-1603`** (`_glass_value_cache`).  A parenthetical
   corrected the comment's OWN earlier wording (*the historical "femtometre"
   wording was wrong -- `round(wavelength * 1e12)` is picometre resolution*).
   The source now states the key in picometres, with the expression, once; the
   correction is in the document.
3. **`lumenairy/raytrace/trace.py:1280`** (`apply_doe_phase_traced` Notes).  The
   docstring corrected an EARLIER VERSION OF ITSELF: *"the pre-R5 docstring's
   claim that it 'neglects the cosine factor that distinguishes sin from the
   direction cosine' was itself inaccurate"*.  The source now states the settled
   fact once -- the direction-cosine form is exact for in-plane diffraction and
   the only approximation is the missing `1 / n2`.

Three further self-correcting notes were relocated rather than corrected,
because the settled statement was already present and only the retraction
needed to go: `raytrace/seidel.py:206` (*"an earlier draft of this fix applied
the sign here and broke seven S11-1 pins; that was a category error,
retracted"*), `raytrace/seidel.py:229` (*"an earlier draft of this note wrongly
called `fnum` defective on that basis; the measurement retracts it"*), and
`io/prescriptions_builders.py:475`, which corrected a measured figure in its own
comment (*"the earlier '199.68' in this comment was wrong in its last two
digits"*) -- the corrected figure and its independent 2x2 ABCD re-measurement
stayed.

---

## 5. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.  The machine was shared
with the other two sweeps and the hygiene agent throughout.

### 5a. The checker and every source-reading walker that touches this partition

```
pytest tests/unit/test_audit2609_a17_history_relocation.py \
       tests/unit/test_v5_3_2_walker_source_line_citation.py \
       tests/unit/test_v4_15_agent_f.py \
       tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
       tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py \
       tests/unit/test_v5_2_walker_pep562_forwarding.py \
       tests/unit/test_v4_16_3_agent_d.py \
       tests/unit/test_v4_16_0_walker_all_symmetry.py \
       tests/unit/test_v5_2_walker_shell_vs_canonical.py \
       tests/unit/test_v4_16_0_walker_sentinel_propagation.py \
       tests/unit/test_v4_16_0_walker_xp_of_dispatch.py \
       tests/unit/test_v5_4_1_walker_scope_the_workaround.py \
       tests/unit/test_audit2609_a9_ui.py
-> 3 failed, 743 passed in 41.85 s
```

That run was taken while the other sweeps were mid-edit; on the **delivered
state** the checker and every walker are green:

```
pytest tests/unit/test_audit2609_a17_history_relocation.py
-> 697 passed in 29.25 s

pytest tests/unit/test_audit2609_a17_history_relocation.py \
       tests/unit/test_v5_3_2_walker_source_line_citation.py \
       tests/unit/test_v4_15_agent_f.py \
       tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
       tests/unit/test_v5_2_walker_pep562_forwarding.py \
       tests/unit/test_v4_16_3_agent_d.py \
       tests/unit/test_v4_16_0_walker_all_symmetry.py \
       tests/unit/test_audit2609_a9_ui.py
-> 793 passed in 34.94 s
```

All 45 of my documents pass both identity tests, the TOC-order test and the
source-pointer test.  `test_v5_3_2_walker_source_line_citation.py` (V18) and
`test_v4_15_agent_f.py` -- the CHANGELOG line-citation pins the brief flagged --
are green, as are `test_v4_16_2_dispatcher_pin_doc_consistency.py` and the
`test_v4_1*_agent_*` / walker files.

Three intermediate failures during the sweep were all transient or upstream and
are all gone on the delivered state: eight `elements/*` identity failures from
another partition's in-flight edits, and one `test_the_fingerprints_are_actually_sensitive[lumenairy._context]`
gap in the checker's mutation 3 (sec. 6).

### 5b. Every test file in `tests/` that reads the source text or docstrings of a module in this partition

The list was built mechanically -- a `getsource` / `getsourcelines` / `getdoc` /
`__doc__` / `read_text` / `getcomments` call on a line that names one of my
modules, PLUS a second sweep for any file that names one of my module
FILENAMES literally (the sweep part 1 added after `test_v4_16_3_agent_d.py`
slipped through the first one).  60 files:

```
pytest <60 files> -q --tb=line
-> 5 failed, 2615 passed, 36 skipped, 54 warnings in 481.86 s (8:01)
```

All five failures are outside this WP:

| failure | why it is not mine |
|---|---|
| `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512-complex128]` and `[1024-complex128]` | `estimate_asm_memory` lives in `lumenairy/memory.py`, held by the hygiene agent; the test's own docstring records WP-A15b lowering `_ASM_FIRST_CALL_FIXED_BYTES` 56 -> 40 MiB TODAY (2026-09-12).  Measured est/cold 0.888 / 0.972 against a `>= 1.0` bound, on a shared box.  Reproduces in isolation; nothing here touches it. |
| `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py::test_no_unguarded_zero_plus_zeroj_in_np_where[propagators/asymptotic_jax_twin.py]` | names `propagators/asymptotic_jax_twin.py:524`; `propagators/` is not in this partition. |
| `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py::test_every_cache_owning_module_enrolls_with_registry` | names `lumenairy/elements/lens_config.py:147 _VOCAB_CACHE`, a WP-A16 file with no `register_cache_clearer` call (`grep -c` = 0) and no working-tree modification. |
| `test_niche_audit_w4_gaps.py::TestW41OddAsphericPowerWaveOptics::test_guard_is_the_shared_checker_not_a_copy` | a transient from another sweep's in-flight edit to `elements/`.  GREEN on the delivered state -- re-run of the whole file: `4 failed, 398 passed`, with this test passing and the other four being the three rows above. |

Confirmation run on the delivered state, of exactly those four files:

```
pytest tests/unit/test_niche_audit_w4_gaps.py \
       tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py \
       tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py \
       tests/unit/test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory
-> 4 failed, 398 passed in 15.22 s
```

-- the same three non-mine failures (one of them twice), stable and reproducing
in isolation, and the `w4_gaps` transient gone.

### 5c. Static gates and import

`ruff check` over every directory in the partition -- *All checks passed*.
`python -c "import lumenairy"` -- OK.  All 233 `lumenairy/**/*.py` compile.
The UI package cannot be imported here (PySide6 absent; the auditor's stub
under `repro/UI/stub` supplies only a 60-line `QtCore`, so `QtWidgets` /
`QtGui` imports still fail), so UI verification is: `py_compile` on all 51 UI
modules, the two fingerprints, `ruff`, and `test_audit2609_a9_ui.py`
(36 passed).

### 5d. A note on method, because it cost a re-run

My apply tool originally read the module with `Path.read_text()`, which applies
universal-newline translation -- so `"\r\n" in src` is always False and the tool
rewrote every CRLF file as LF.  `git diff` stayed clean (this checkout is
`core.autocrlf=true`, so the blob is LF either way) but the working tree
diverged from the checkout convention on 12 files.  Fixed (`read_bytes`), and
the 12 files were restored to CRLF.  A second bug in the same tool: it
fingerprinted the CRLF text while the checker fingerprints the
universal-newline text, so a CRLF module carrying a multi-line NON-docstring
string literal (`ui/main_window.py`) was wrongly refused.  Both are fixed, and
both failed SAFE -- the first produced no content change, the second refused to
write.

---

## 6. Modules deliberately left alone, and the one checker gap

**17 of the 111 modules were not changed at all.**  Fifteen carry no history
after the census:

```
lumenairy/_math/__init__.py          lumenairy/ui/_mpl.py
lumenairy/_math/levin.py             lumenairy/ui/_worker.py
lumenairy/backend/_optional.py       lumenairy/ui/analysis.py
lumenairy/io/__init__.py             lumenairy/ui/command_palette.py
lumenairy/raytrace/_conic_core.py    lumenairy/ui/diagnostics.py
lumenairy/raytrace/exit_vertex.py    lumenairy/ui/layout_3d.py
lumenairy/raytrace/world.py          lumenairy/ui/surface_editors.py
                                     lumenairy/ui/workspace.py
```

The other two were left alone ON PURPOSE:

* **`lumenairy/optimize/core.py`** and **`lumenairy/raytrace/core.py`** -- the
  source-grep contract described in sec. 1.3.

A third module, **`lumenairy/io/prescriptions_code_v.py`**, kept its entire
`pre-v5.46` narrative for the file-format reason in sec. 1.3; only three bare
release tags were stripped from it.  It is the one module in the partition
whose strict count barely moves (19 -> 16), and that is the intended outcome.

`lumenairy/memory.py`, `lumenairy/__init__.py`,
`lumenairy/ui/lens_options_dialog.py` and `lumenairy/backend/__init__.py` are
outside the partition and were not touched.  **The partition is complete:
111 modules enumerated, 94 changed, 17 deliberately untouched, 0 skipped.**

### Three checker gaps this sweep surfaced -- all now fixed by the orchestrator

Extending the registry from 3 documents to 111 exercised three shapes the
part-1 checker had not met.  All three were hit here, reported, and fixed in
`tests/unit/test_audit2609_a17_history_relocation.py` while the sweep ran
(I did not edit that file -- it is WP-A17 part 1 / orchestrator territory):

| gap | what triggered it | fix that landed |
|---|---|---|
| the document-name assertion rejected dotted names | every document in this sweep (`lumenairy.io.storage`) | derive `dotted` from the header's repo-relative `module:` instead of from the absolute `src_path` -- the old form produced `D:\.Metacept.…lumenairy.io.storage` |
| mutation 1: "no single-line statement found to delete" | a module whose only multi-statement function ends in a multi-line `return {...}` (`analysis/coronagraph.py`, `optimize/multi_objective.py`) | search every body statement from the end, skipping docstring statements, instead of only the last one |
| mutation 3: "no small integer literal found to re-spell" | **`lumenairy/_context.py`, which contains ZERO integer constants** (verified by walking its AST: 0 `ast.Constant` nodes of type `int`), so `5 -> 0x5` had nothing to bite on | try an integer first, then a string's quote style -- also value-preserving, also token-only |

Two modules in the partition are structurally unable to satisfy mutation 1 at
all (pure re-export shells with no `FunctionDef`): `optimize/__init__.py` and
`io/prescriptions.py`.  Both carried nothing but a release tag and a
split-provenance sentence, so both were handled as tag-only strips (sec. 8) and
neither got a document -- the narrative still leaves the source, and the
checker's registry does not have to grow a case it cannot mutate.

Two modules in the partition are structurally unable to satisfy mutation 1
(pure re-export shells with no `FunctionDef` at all): `optimize/__init__.py`
and `io/prescriptions.py`.  Both had a release tag and a split-provenance
sentence and nothing else, so both were handled as tag-only strips (sec. 8) and
neither got a document -- which keeps the checker green on them without leaving
the narrative in the source.

---

## 7. Requested changes outside my ownership

1. ~~**`tests/unit/test_audit2609_a17_history_relocation.py`, mutation 3**~~ --
   **RESOLVED while this sweep ran.**  The checker now tries an integer
   re-spelling first and a string's quote style where the module carries no
   integer, which is what `_context.py` needed.  Recorded here because the
   three checker gaps in sec. 6 are all consequences of growing the registry
   from 3 documents to 111, and a fourth sweep would want to know that.

2. **`CHANGELOG.md` line-citation refresh** (documentation accuracy; no test
   currently fails on it).  `optimize/wrapper_merits.py:987` is cited for the
   `_ZERO_APERTURE_MASK` sentinel branch; that branch now sits at **line 958**.
   `test_v4_15_agent_f.py::test_changelog_cites_actual_current_lines` stays
   GREEN only because the CHANGELOG also carries an older
   `optimize/wrapper_merits.py:955` citation which happens to land inside the
   +/- 5 window.  The `:987` citation is stale by 29 lines and should read
   `:958`.  For the sibling anchor I took the other route: every edit ABOVE
   `merit_terms.py`'s `np.isfinite(ap) and ap > 0` line is line-count neutral
   on purpose, so that anchor has not moved and its `:638` citation needs no
   refresh.

3. **`lumenairy/elements/lens_config.py`** (WP-A16) -- `_VOCAB_CACHE` at line
   147 is not enrolled with the central cache registry, so
   `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py::test_every_cache_owning_module_enrolls_with_registry`
   is red.  Either add the module-level `register_cache_clearer(...)` the test's
   own message spells out, or add the exemption with a cited rationale.
   Reported only; the file is not mine.

---

## 8. Appendix -- the 287 tag-only strips, verbatim

A release/audit TAG on an otherwise-live why-comment was stripped in place
rather than relocated into a document (sec. 1.2c).  Nothing but the tag was
removed: the remainder of every line is preserved character-for-character
except for capitalising its first letter.  Identical texts are grouped, with
one example site and the total count.

| n | tag stripped (before) | left in the source (after) | site(s) |
|---|---|---|---|
| 38 | `"""v5.4.4 (audit GUI-resize round 2): report a tiny minimum so` | `"""Report a tiny minimum so` | 38 sites, e.g. `lumenairy/ui/algebra_dock.py:834` |
| 38 | `"""v5.4.4: companion to minimumSizeHint() above.  Provides a` | `"""Companion to minimumSizeHint() above.  Provides a` | 38 sites, e.g. `lumenairy/ui/algebra_dock.py:846` |
| 35 | `# v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink` | `# Override matplotlib canvas sizeHint so the dock can shrink` | 35 sites, e.g. `lumenairy/ui/algebra_dock.py:354` |
| 3 | `# v5.4 (audit P1-F): cooperative cancel via CancellableProgress.` | `# Cooperative cancel via CancellableProgress.` | 3 sites, e.g. `lumenairy/ui/multiconfig_dock.py:269` |
| 3 | `# v5.4 (audit P1-F): wire CancellableProgress + Stop button` | `# Wire CancellableProgress + Stop button` | 3 sites, e.g. `lumenairy/ui/multiconfig_dock.py:11` |
| 2 | `# v4.15.1 (P3-2 / Agent E): upper-bound major-version-bump warning.` | `# Upper-bound major-version-bump warning.` | 2 sites, e.g. `lumenairy/io/codegen.py:779` |
| 2 | `# v4.16.0: acquire the cross-process append lock BEFORE opening` | `# Acquire the cross-process append lock BEFORE opening` | 2 sites, e.g. `lumenairy/io/storage.py:959` |
| 2 | `# v5.4 (audit P1-F): Stop button -- cooperative cancel via` | `# Stop button -- cooperative cancel via` | 2 sites, e.g. `lumenairy/ui/multiconfig_dock.py:151` |
| 1 | `"""v4.15 (P1-UI-4): world-coords back vertex of ``prev`` element.` | `"""World-coords back vertex of ``prev`` element.` | `lumenairy/ui/model.py:908` |
| 1 | `"""v5.4 (audit P1-D): NSGA-II Pareto front via pymoo.` | `"""NSGA-II Pareto front via pymoo.` | `lumenairy/ui/optimizer_dock.py:1417` |
| 1 | `"""v5.4 (audit P1-D): pre-flight check on dock-supplied kwargs.` | `"""Pre-flight check on dock-supplied kwargs.` | `lumenairy/ui/optimizer_dock.py:1372` |
| 1 | `"""v5.4.2 (audit C1): wait briefly for an in-flight worker` | `"""Wait briefly for an in-flight worker` | `lumenairy/ui/coronagraph_dock.py:712` |
| 1 | `"""v5.4.2: deferred system_changed emit slot."""` | `"""Deferred system_changed emit slot."""` | `lumenairy/ui/slider_dock.py:346` |
| 1 | `"""v5.4.6 (audit F-18): build a uniform circular pupil and call the` | `"""Build a uniform circular pupil and call the` | `lumenairy/ui/richards_wolf_dock.py:39` |
| 1 | `# v4.11.2: no mirror-parity flip -- ``thickness_m`` is` | `# No mirror-parity flip -- ``thickness_m`` is` | `lumenairy/io/prescriptions_zemax.py:2461` |
| 1 | `# v4.12.1: detect the pure-spherical fast path.  Requires` | `# Detect the pure-spherical fast path.  Requires` | `lumenairy/raytrace/intersection.py:172` |
| 1 | `# v4.12.2: converted to an LRU-bounded ``OrderedDict`` so long-running` | `# Converted to an LRU-bounded ``OrderedDict`` so long-running` | `lumenairy/raytrace/jax_trace.py:1144` |
| 1 | `# v4.13.1 (P1-F): honour replace=False on the JAX backend.` | `# Honour replace=False on the JAX backend.` | `lumenairy/backend/random.py:166` |
| 1 | `# v4.13.2 (C-P0-4): also capture the optional SI image-plane block` | `# Also capture the optional SI image-plane block` | `lumenairy/io/prescriptions_code_v.py:502` |
| 1 | `# v4.13.2 (C-P0-4): preserve the BFL.  Prefer the SI image-plane` | `# Preserve the BFL.  Prefer the SI image-plane` | `lumenairy/io/prescriptions_code_v.py:904` |
| 1 | `# v4.13.2 (C-P0-5): capture the last surface's THI as the` | `# Capture the last surface's THI as the` | `lumenairy/io/prescriptions_quadoa.py:347` |
| 1 | `# v4.13.2 (C-P0-5): preserve the BFL.  Prefer the top-level` | `# Preserve the BFL.  Prefer the top-level` | `lumenairy/io/prescriptions_quadoa.py:397` |
| 1 | `# v4.13.2 (C-P0-5): track the last surface's THI as a BFL` | `# Track the last surface's THI as a BFL` | `lumenairy/io/prescriptions_quadoa.py:309` |
| 1 | `# v4.13.2 (P1-NEW-I): pin the int dtype to int64 so the JAX` | `# Pin the int dtype to int64 so the JAX` | `lumenairy/backend/random.py:188` |
| 1 | `# v4.13.2 (P1-NEW-K): wrap the replace=False dispatch in` | `# Wrap the replace=False dispatch in` | `lumenairy/backend/random.py:174` |
| 1 | `# v4.13.2 (audit P1-NEW-J): clone the last surface with the new` | `# Clone the last surface with the new` | `lumenairy/raytrace/trace.py:1473` |
| 1 | `# v4.15 (P1-CG): same version-pin rationale as _generate_unrolled.` | `# Same version-pin rationale as _generate_unrolled.` | `lumenairy/io/codegen.py:1058` |
| 1 | `# v4.15 (P1-CG): stamp the generating library version into the` | `# Stamp the generating library version into the` | `lumenairy/io/codegen.py:730` |
| 1 | `# v4.15 (P1-UI-1): high-index glass table for the` | `# High-index glass table for the` | `lumenairy/ui/main_window.py:1658` |
| 1 | `# v4.15 (P1-UI-3): restore the three fields that v4.14.x` | `# Restore the three fields that v4.14.x` | `lumenairy/ui/model.py:1648` |
| 1 | `# v4.15 (P1-UI-4): route both call sites through one helper so` | `# Route both call sites through one helper so` | `lumenairy/ui/model.py:874` |
| 1 | `# v4.15 (P1-UI-5): guard against the parent's C++ side` | `# Guard against the parent's C++ side` | `lumenairy/ui/waveoptics_dock.py:2707` |
| 1 | `# v4.15 (P1-UI-6): accumulate mean OPD per pixel rather than` | `# Accumulate mean OPD per pixel rather than` | `lumenairy/ui/psf_mtf_dock.py:272` |
| 1 | `# v4.15 (P1-UI-7): bounds-mask the rays BEFORE indexing so any` | `# Bounds-mask the rays BEFORE indexing so any` | `lumenairy/ui/psf_mtf_dock.py:258` |
| 1 | `# v4.15.0 (P2-CTX-1): a single call to` | `# A single call to` | `lumenairy/_context.py:258` |
| 1 | `# v4.15.1 (Cluster B Item 6): bridge a coherent field into a RayBundle.` | `# Bridge a coherent field into a RayBundle.` | `lumenairy/raytrace/__init__.py:89` |
| 1 | `# v4.15.1 (P1-NEW-C / Agent E): migrated to the canonical` | `# Migrated to the canonical` | `lumenairy/ui/model.py:276` |
| 1 | `# v4.15.2 (audit P1-NEW-A): invoke the literal 3-stage chain` | `# Invoke the literal 3-stage chain` | `lumenairy/algebra/primitives.py:798` |
| 1 | `# v4.15.2 (audit P1-NEW-C): thread ``dy`` into the dispatcher` | `# Thread ``dy`` into the dispatcher` | `lumenairy/algebra/primitives.py:293` |
| 1 | `# v4.15.2: the propagator backend for the two FreeSpace legs` | `# The propagator backend for the two FreeSpace legs` | `lumenairy/algebra/primitives.py:774` |
| 1 | `# v4.15.3 (audit P1-NEW-F1-1): the two FreeSpace legs below` | `# The two FreeSpace legs below` | `lumenairy/algebra/primitives.py:807` |
| 1 | `# v4.15.3 (audit P1-NEW-F1-1): when ``dy != dx`` (anamorphic` | `# When ``dy != dx`` (anamorphic` | `lumenairy/algebra/primitives.py:308` |
| 1 | `# v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared` | `# Defensive guard via the shared` | `lumenairy/raytrace/from_field.py:277` |
| 1 | `# v4.15: dtype-aware sentinel migration.  Pre-4.15 the` | `# Dtype-aware sentinel migration.  Pre-4.15 the` | `lumenairy/ui/psf_mtf_dock.py:302` |
| 1 | `# v4.16 (ROADMAP #11): multi-objective Pareto via pymoo NSGA-II.` | `# Multi-objective Pareto via pymoo NSGA-II.` | `lumenairy/optimize/__init__.py:62` |
| 1 | `# v4.16.0 (Agent A __all__-symmetry walker): pymoo availability` | `# Pymoo availability` | `lumenairy/optimize/__init__.py:68` |
| 1 | `# v4.16.0: distributed (multi-process) advisory lock.  Imported lazily` | `# Distributed (multi-process) advisory lock.  Imported lazily` | `lumenairy/io/storage.py:94` |
| 1 | `# v4.16.0: forward lock_timeout to the zarr path; drop` | `# Forward lock_timeout to the zarr path; drop` | `lumenairy/io/storage.py:1828` |
| 1 | `# v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the prior` | `# The prior` | `lumenairy/_validation.py:171` |
| 1 | `# v5.0.1 (audit P1-NEW-V4-1): TYPE_CHECKING guard for forward reference` | `# TYPE_CHECKING guard for forward reference` | `lumenairy/algebra/base.py:36` |
| 1 | `# v5.17.1 (audit P1-07): the name was id(elem)-derived, but` | `# The name was id(elem)-derived, but` | `lumenairy/raytrace/trace.py:1649` |
| 1 | `# v5.17.1 (audit P2-36): the '__user__' sentinel is what` | `# The '__user__' sentinel is what` | `lumenairy/raytrace/trace.py:1754` |
| 1 | `# v5.17.1 (audit P2-41): re-pointing the registry must also drop` | `# Re-pointing the registry must also drop` | `lumenairy/user_library.py:518` |
| 1 | `# v5.18.1: the dock stores the model as ``self.sm`` (see` | `# The dock stores the model as ``self.sm`` (see` | `lumenairy/ui/waveoptics_dock.py:3164` |
| 1 | `# v5.2 (AUDIT_V4_13_1 P1-G closure): preserve the sign of the` | `# Preserve the sign of the` | `lumenairy/raytrace/trace.py:1339` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): annotate the` | `# Annotate the` | `lumenairy/backend/random.py:51` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): fft_infra is` | `# ``fft_infra`` is` | `lumenairy/backend/fft.py:125` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): narrow the` | `# Narrow the` | `lumenairy/progress.py:153` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): the backend` | `# The backend` | `lumenairy/backend/scipy.py:199` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): the public` | `# The public` | `lumenairy/backend/random.py:148` |
| 1 | `# v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure): the type` | `# The type` | `lumenairy/backend/array.py:209` |
| 1 | `# v5.2.5 (AUDIT_V5_2_3 P3-F4): legacy public name preserved as a` | `# Legacy public name preserved as a` | `lumenairy/_context.py:358` |
| 1 | `# v5.2.5 (AUDIT_V5_2_3 P3-F4): the public name ``install_atexit_restore``` | `# The public name ``install_atexit_restore``` | `lumenairy/_context.py:46` |
| 1 | `# v5.24.3 (audit S4-2): the bounds' centre must be in the` | `# The bounds' centre must be in the` | `lumenairy/ui/optimizer_dock.py:1102` |
| 1 | `# v5.24.4 (audit S4-7): ``merit_function`` mutated the live` | `# ``merit_function`` mutated the live` | `lumenairy/ui/model.py:3124` |
| 1 | `# v5.24.4 (audit S4-7): apply_result=False -- the model runs` | `# Apply_result=False -- the model runs` | `lumenairy/ui/optimizer_dock.py:177` |
| 1 | `# v5.24.4 (audit S4-7): the worker runs the optimization with` | `# The worker runs the optimization with` | `lumenairy/ui/optimizer_dock.py:102` |
| 1 | `# v5.24.4 (audit S4-7, part 1): ``apply_result`` gates the` | `# ``apply_result`` gates the` | `lumenairy/ui/model.py:3079` |
| 1 | `# v5.24.4 (audit S4-7, part 2): the geometric model stores NO` | `# The geometric model stores NO` | `lumenairy/ui/model.py:3067` |
| 1 | `# v5.24.x (audit S4-19): even the empty-selection path honours a` | `# Even the empty-selection path honours a` | `lumenairy/io/storage.py:2025` |
| 1 | `# v5.24.x (audit S4-19): warn on a GENUINELY malformed` | `# Warn on a GENUINELY malformed` | `lumenairy/io/prescriptions_zemax.py:1671` |
| 1 | `# v5.24.x (audit S4-20): use the CONJUGATE phase convention` | `# Use the CONJUGATE phase convention` | `lumenairy/ui/psf_mtf_dock.py:294` |
| 1 | `# v5.24.x (audit S4-6): (elem_idx, surf_idx) -> internal-gap` | `# (elem_idx, surf_idx) -> internal-gap` | `lumenairy/ui/optimizer_dock.py:1005` |
| 1 | `# v5.24.x (audit S4-6): a LAST-surface ``thickness`` is the air` | `# A LAST-surface ``thickness`` is the air` | `lumenairy/ui/optimizer_dock.py:1031` |
| 1 | `# v5.24.x (audit S4-6): a last-surface thickness and the` | `# A last-surface thickness and the` | `lumenairy/ui/optimizer_dock.py:1092` |
| 1 | `# v5.24.x (audit S4-6): glass / semi_diameter etc. have` | `# Glass / semi_diameter etc. have` | `lumenairy/ui/optimizer_dock.py:1086` |
| 1 | `# v5.24.x (audit S4-6): route a surface thickness to its` | `# Route a surface thickness to its` | `lumenairy/ui/optimizer_dock.py:1066` |
| 1 | `# v5.24.x (audit S4-9): a prescription with no STOP and no semi-diameter` | `# A prescription with no STOP and no semi-diameter` | `lumenairy/io/prescriptions_zemax.py:1235` |
| 1 | `# v5.24.x (audit S4-9): an unrecognised UNIT token silently` | `# An unrecognised UNIT token silently` | `lumenairy/io/prescriptions_zemax.py:633` |
| 1 | `# v5.24.x (audit S4-9): expose the explicit stop index at the top level` | `# Expose the explicit stop index at the top level` | `lumenairy/io/prescriptions_zemax.py:1961` |
| 1 | `# v5.24.x (audit S4-9): mirror the .zmx twin (ZX-3) and carry` | `# Mirror the .zmx twin (ZX-3) and carry` | `lumenairy/io/prescriptions_zemax.py:1920` |
| 1 | `# v5.24.x (audit S4-9): warn before the silent mm fallback` | `# Warn before the silent mm fallback` | `lumenairy/io/prescriptions_zemax.py:1576` |
| 1 | `# v5.24.x (audit S4-9): warn on a silent 0.0 aperture (no STOP + no` | `# Warn on a silent 0.0 aperture (no STOP + no` | `lumenairy/io/prescriptions_zemax.py:1897` |
| 1 | `# v5.25 (audit S3-16): emit the canonical ``w0`` waist (1/e^2 intensity` | `# Emit the canonical ``w0`` waist (1/e^2 intensity` | `lumenairy/io/codegen.py:809` |
| 1 | `# v5.3 (AUDIT_V5_2_5 P3-5): the ``if max_k >= 1`` guard above` | `# The ``if max_k >= 1`` guard above` | `lumenairy/_math/chebyshev.py:156` |
| 1 | `# v5.30 (audit P5 / roadmap F1, flip-day migration): ``return_result``` | `# ``return_result``` | `lumenairy/algebra/primitives.py:334` |
| 1 | `# v5.30: ``AberrationTensorResult`` carries the matrix on` | `# ``AberrationTensorResult`` carries the matrix on` | `lumenairy/ui/lg_aberration_dock.py:141` |
| 1 | `# v5.30: label rows by their (p, ell) OUTPUT mode -- the` | `# Label rows by their (p, ell) OUTPUT mode -- the` | `lumenairy/ui/lg_aberration_dock.py:165` |
| 1 | `# v5.31 (audit A-9): the closed vocabulary for ``input_kind``.` | `# The closed vocabulary for ``input_kind``.` | `lumenairy/_validation.py:47` |
| 1 | `# v5.4 (audit P1-B): surface ``plot_wavefront()`` (v4.14.0) in the` | `# Surface ``plot_wavefront()`` (v4.14.0) in the` | `lumenairy/ui/wavefront_map_dock.py:4` |
| 1 | `# v5.4 (audit P1-D): "Advanced parameters" collapsible group` | `# "Advanced parameters" collapsible group` | `lumenairy/ui/optimizer_dock.py:420` |
| 1 | `# v5.4 (audit P1-D): Advanced-parameters group.` | `# Advanced-parameters group.` | `lumenairy/ui/optimizer_dock.py:580` |
| 1 | `# v5.4 (audit P1-D): Constraint editor sub-panel.` | `# Constraint editor sub-panel.` | `lumenairy/ui/optimizer_dock.py:1867` |
| 1 | `# v5.4 (audit P1-D): NSGA-II Pareto front branch.` | `# NSGA-II Pareto front branch.` | `lumenairy/ui/optimizer_dock.py:1468` |
| 1 | `# v5.4 (audit P1-D): ``method`` kwarg surfaces the optimizer-` | `# ``method`` kwarg surfaces the optimizer-` | `lumenairy/ui/model.py:3061` |
| 1 | `# v5.4 (audit P1-D): canonical scipy / design_optimize method tokens` | `# Canonical scipy / design_optimize method tokens` | `lumenairy/ui/optimizer_dock.py:28` |
| 1 | `# v5.4 (audit P1-D): dock-supplied advanced parameter dict.` | `# Dock-supplied advanced parameter dict.` | `lumenairy/ui/optimizer_dock.py:93` |
| 1 | `# v5.4 (audit P1-D): forward the Advanced-parameters dock` | `# Forward the Advanced-parameters dock` | `lumenairy/ui/optimizer_dock.py:861` |
| 1 | `# v5.4 (audit P1-D): forward the full Advanced-parameters` | `# Forward the full Advanced-parameters` | `lumenairy/ui/optimizer_dock.py:1244` |
| 1 | `# v5.4 (audit P1-D): full advanced-parameter dict from the` | `# Full advanced-parameter dict from the` | `lumenairy/ui/optimizer_dock.py:1353` |
| 1 | `# v5.4 (audit P1-D): merge dock-supplied advanced kwargs` | `# Merge dock-supplied advanced kwargs` | `lumenairy/ui/optimizer_dock.py:1495` |
| 1 | `# v5.4 (audit P1-D): parameter surface expansion for v4.16.0 optimisation framework` | `# Parameter surface expansion for v4.16.0 optimisation framework` | `lumenairy/ui/optimizer_dock.py:5` |
| 1 | `# v5.4 (audit P1-D): validate dock kwarg combinations BEFORE` | `# Validate dock kwarg combinations BEFORE` | `lumenairy/ui/optimizer_dock.py:135` |
| 1 | `# v5.4 (audit P1-E): expand from 41-LOC stub to full algorithm-dispatched dock` | `# Expand from 41-LOC stub to full algorithm-dispatched dock` | `lumenairy/ui/phase_retrieval_dock.py:1` |
| 1 | `# v5.4 (audit P1-F): CancellableProgress wraps the existing` | `# CancellableProgress wraps the existing` | `lumenairy/ui/optimizer_dock.py:1360` |
| 1 | `# v5.4 (audit P1-F): also note user cancellation so the summary` | `# Also note user cancellation so the summary` | `lumenairy/ui/phase_retrieval_dock.py:837` |
| 1 | `# v5.4 (audit P1-F): also reset UI on cooperative cancel.` | `# Also reset UI on cooperative cancel.` | `lumenairy/ui/optimizer_dock.py:870` |
| 1 | `# v5.4 (audit P1-F): cancellation flag polled by the scipy` | `# Cancellation flag polled by the scipy` | `lumenairy/ui/optimizer_dock.py:97` |
| 1 | `# v5.4 (audit P1-F): cooperative cancellation -- workers poll` | `# Cooperative cancellation -- workers poll` | `lumenairy/ui/optimizer_dock.py:876` |
| 1 | `# v5.4 (audit P1-F): cooperative-cancel via the library protocol.` | `# Cooperative-cancel via the library protocol.` | `lumenairy/ui/phase_retrieval_dock.py:40` |
| 1 | `# v5.4 (audit P1-F): emit cancelled before finished so the` | `# Emit cancelled before finished so the` | `lumenairy/ui/tolerance_dock.py:179` |
| 1 | `# v5.4 (audit P1-F): if user cancelled, emit cancelled` | `# If user cancelled, emit cancelled` | `lumenairy/ui/phase_retrieval_dock.py:233` |
| 1 | `# v5.4 (audit P1-F): inform the summary on user cancel;` | `# Inform the summary on user cancel;` | `lumenairy/ui/multiconfig_dock.py:256` |
| 1 | `# v5.4 (audit P1-F): map cancel signal to the same UI reset.` | `# Map cancel signal to the same UI reset.` | `lumenairy/ui/optimizer_dock.py:967` |
| 1 | `# v5.4 (audit P1-F): partial-result still flushed via` | `# Partial-result still flushed via` | `lumenairy/ui/tolerance_dock.py:343` |
| 1 | `# v5.4 (audit P1-F): poll for cooperative cancel between` | `# Poll for cooperative cancel between` | `lumenairy/ui/tolerance_dock.py:84` |
| 1 | `# v5.4 (audit P1-F): polled between restarts (and inside each` | `# Polled between restarts (and inside each` | `lumenairy/ui/optimizer_dock.py:1562` |
| 1 | `# v5.4 (audit P1-F): polled between trials in the run() loop.` | `# Polled between trials in the run() loop.` | `lumenairy/ui/tolerance_dock.py:57` |
| 1 | `# v5.4 (audit P1-F): replace ad-hoc ``_stop_requested`` flag` | `# Replace ad-hoc ``_stop_requested`` flag` | `lumenairy/ui/phase_retrieval_dock.py:203` |
| 1 | `# v5.4 (audit P1-F): wave worker emits its own cancelled` | `# Wave worker emits its own cancelled` | `lumenairy/ui/optimizer_dock.py:1270` |
| 1 | `# v5.4 (audit P1-F): wraps the existing Qt-emit callback.` | `# Wraps the existing Qt-emit callback.` | `lumenairy/ui/multiconfig_dock.py:46` |
| 1 | `# v5.4 (audit P2-C): expand from 141-LOC single-function wrapper` | `# Expand from 141-LOC single-function wrapper` | `lumenairy/ui/ghost_dock.py:1` |
| 1 | `# v5.4 (audit P2-E): Stokes and polarisation-derived tabs` | `# Stokes and polarisation-derived tabs` | `lumenairy/ui/jones_pupil_dock.py:337` |
| 1 | `# v5.4 (audit P2-E): add Stokes + DOP tabs` | `# Add Stokes + DOP tabs` | `lumenairy/ui/jones_pupil_dock.py:14` |
| 1 | `# v5.4 (audit P3-A): expand from 162-LOC Schell-source-only to 4 tabs` | `# Expand from 162-LOC Schell-source-only to 4 tabs` | `lumenairy/ui/coherence_dock.py:1` |
| 1 | `# v5.4.1 (audit P3 #8): tighten prune threshold from 1e-15 (ULP` | `# Tighten prune threshold from 1e-15 (ULP` | `lumenairy/_math/chebyshev.py:416` |
| 1 | `# v5.4.2 (audit B-P2 belt-and-suspenders): inner 40 ms` | `# Inner 40 ms` | `lumenairy/ui/slider_dock.py:145` |
| 1 | `# v5.4.2 (post-v5.4.1 user-reported GUI hang): build the` | `# Build the` | `lumenairy/ui/model.py:2725` |
| 1 | `# v5.4.2: emit system_changed via 40 ms inner debounce` | `# Emit system_changed via 40 ms inner debounce` | `lumenairy/ui/slider_dock.py:337` |
| 1 | `# v5.4.6 (audit F-14): the prescription stores the mirror radius` | `# The prescription stores the mirror radius` | `lumenairy/io/codegen.py:950` |
| 1 | `# v5.4.6 (audit F-15): preserve the propagation distances INTO` | `# Preserve the propagation distances INTO` | `lumenairy/io/prescriptions_transforms.py:583` |
| 1 | `# v5.4.6 (audit F-16): validate 0 <= inner < D FIRST, then apply` | `# Validate 0 <= inner < D FIRST, then apply` | `lumenairy/algebra/apertures.py:88` |
| 1 | `# v5.4.6 (audit F-17): apply the new state INSIDE a try so that a` | `# Apply the new state INSIDE a try so that a` | `lumenairy/_context.py:238` |
| 1 | `# v5.4.6 (audit F-29): default stop_surface to the prescription's own` | `# Default stop_surface to the prescription's own` | `lumenairy/io/prescriptions_code_v.py:208` |
| 1 | `# v5.4.6 (audit F-29): when the caller does not pass stop_surface` | `# When the caller does not pass stop_surface` | `lumenairy/io/prescriptions_quadoa.py:169` |
| 1 | `# v5.4.6 (audit F-30): match the NumPy/CuPy int64 default on` | `# Match the NumPy/CuPy int64 default on` | `lumenairy/backend/random.py:143` |
| 1 | `# v5.4.6 (audit F-31): match the NumPy/CuPy default precision.` | `# Match the NumPy/CuPy default precision.` | `lumenairy/backend/random.py:100` |
| 1 | `# v5.4.6 (audit F-31): x64-aware default; see ``uniform``.` | `# x64-aware default; see ``uniform``.` | `lumenairy/backend/random.py:123` |
| 1 | `# v5.4.6 (audit F-35): warn before overriding a built-in registry` | `# Warn before overriding a built-in registry` | `lumenairy/user_library.py:506` |
| 1 | `# v5.4.6 (audit P3-2): outside the conic domain ((1+k)h^2/R^2 >= 1)` | `# Outside the conic domain ((1+k)h^2/R^2 >= 1)` | `lumenairy/raytrace/surface.py:634` |
| 1 | `# v5.4.6 (audit P3-3): keep the tangent case (disc == 0); only` | `# Keep the tangent case (disc == 0); only` | `lumenairy/raytrace/intersection.py:415` |
| 1 | `# v5.4.7 (audit AUDIT_V5_4_6 gap #1): negate the mirror radius` | `# Negate the mirror radius` | `lumenairy/io/codegen.py:1165` |
| 1 | `# v5.4.7 (audit AUDIT_V5_4_6 gap #2): NaN (not 0.0) out of the` | `# NaN (not 0.0) out of the` | `lumenairy/raytrace/surface.py:583` |
| 1 | `# v5.45.2 (audit 2026-09-11 R2 / WP-A7 section 5.3): the functional form` | `# The functional form` | `lumenairy/raytrace/__init__.py:163` |
| 1 | `# v5.46 (audit Z3 / VERIFY-A11 O-1): arm the far-field truncation` | `# Arm the far-field truncation` | `lumenairy/algebra/primitives.py:328` |
| 1 | `# v5.46 (audit Z3): that reason applies ONLY to the anamorphic branch,` | `# That reason applies ONLY to the anamorphic branch,` | `lumenairy/algebra/primitives.py:348` |
| 1 | `# v5.46 (audit Z4 nit): relative, matching the rest of the package -- an` | `# Relative, matching the rest of the package -- an` | `lumenairy/_validation.py:41` |
| 1 | `# v5.46 (audit Z4): make ``lumenairy.algebra.from_prescription`` resolve to` | `# Make ``lumenairy.algebra.from_prescription`` resolve to` | `lumenairy/algebra/__init__.py:92` |
| 1 | `# v5.46 (audit Z4): the numeric ``dtype.kind`` letters this guard accepts.` | `# The numeric ``dtype.kind`` letters this guard accepts.` | `lumenairy/_validation.py:67` |
| 1 | `# v5.4: surface any WFS-fallback note (pyramid/curvature` | `# Surface any WFS-fallback note (pyramid/curvature` | `lumenairy/ui/ao_dock.py:218` |
| 1 | `# v5.4: surface any WFS-fallback note from the worker (e.g. the` | `# Surface any WFS-fallback note from the worker (e.g. the` | `lumenairy/ui/ao_dock.py:927` |
| 1 | `# v5.6: a re-registered name must not serve a stale value from the` | `# A re-registered name must not serve a stale value from the` | `lumenairy/user_library.py:643` |
| 1 | `v4.11.2: the previous 3.6.1-hotfix-6 mirror-parity sign flip was` | `The previous 3.6.1-hotfix-6 mirror-parity sign flip was` | `lumenairy/io/prescriptions_zemax.py:2315` |
| 1 | `v4.12.1 (Track C): pure-spherical surfaces (``conic == 0``, no` | `pure-spherical surfaces (``conic == 0``, no` | `lumenairy/raytrace/intersection.py:143` |
| 1 | `v4.13.1 (P3 #20): extended to recognise opaque keys.  The legacy` | `Extended to recognise opaque keys.  The legacy` | `lumenairy/backend/random.py:202` |
| 1 | `v4.15 (P1-UI-2): routed through ``set_display_distance`` so the` | `Routed through ``set_display_distance`` so the` | `lumenairy/ui/main_window.py:1756` |
| 1 | `v4.15 (P1-UI-3): added ``wavelength_weights``, ``field_weights``,` | `Added ``wavelength_weights``, ``field_weights``,` | `lumenairy/ui/model.py:1609` |
| 1 | `v4.15 (P1-UI-5): the post-exec re-parent back to the original` | `The post-exec re-parent back to the original` | `lumenairy/ui/waveoptics_dock.py:2663` |
| 1 | `v4.15.2 (P3 from the v4.15.1 audit): when ``n_rays`` exceeds the` | `When ``n_rays`` exceeds the` | `lumenairy/raytrace/from_field.py:626` |
| 1 | `v5.21.5 (AUDIT_RAYTRACE_CORE residual): when True, interleave` | `When True, interleave` | `lumenairy/raytrace/trace.py:497` |
| 1 | `v5.24.x (audit S4-19): ``inv_scale`` (the length-scale factor that` | ```inv_scale`` (the length-scale factor that` | `lumenairy/io/prescriptions_quadoa.py:95` |
| 1 | `v5.24.x (audit S4-19): unit-rescale each coefficient.  The library` | `unit-rescale each coefficient.  The library` | `lumenairy/io/prescriptions_quadoa.py:53` |
| 1 | `v5.29.1 (audit P13-P16): this function forwards` | `This function forwards` | `lumenairy/backend/fft.py:91` |
| 1 | `v5.3.2 (ROADMAP "logging adoption sweep"): adds per-iteration` | `Adds per-iteration` | `lumenairy/_logging.py:3` |
| 1 | `v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A): this` | `This` | `lumenairy/ui/__init__.py:15` |
| 1 | `v5.4 (audit P1-D): accepts an ``advanced_kwargs`` dict from the` | `Accepts an ``advanced_kwargs`` dict from the` | `lumenairy/ui/optimizer_dock.py:64` |
| 1 | `v5.4 (audit P1-D): surfaces 8 design_optimize() kwargs that` | `Surfaces 8 design_optimize() kwargs that` | `lumenairy/ui/optimizer_dock.py:586` |
| 1 | `v5.4.2 (audit B-P2 belt-and-suspenders): defer the` | `Defer the` | `lumenairy/ui/slider_dock.py:323` |
| 1 | `v5.4.2 (audit C2 belt-and-suspenders): track that we've` | `Track that we've` | `lumenairy/ui/main_window.py:2788` |
| 1 | `v5.4.2 (audit C2 belt-and-suspenders): wrap restore in` | `Wrap restore in` | `lumenairy/ui/main_window.py:2803` |
| 1 | `v5.4.6 (audit F-18): ``richards_wolf_focus`` returns a TUPLE` | ```richards_wolf_focus`` returns a TUPLE` | `lumenairy/ui/richards_wolf_dock.py:30` |

**286 tag strips across 68 modules; 171 distinct texts.**

---

## 9. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP3_CHANGELOG.md`

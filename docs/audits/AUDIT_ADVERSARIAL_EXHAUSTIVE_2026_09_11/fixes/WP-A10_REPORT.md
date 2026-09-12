# WP-A10 — I/O and optimisation (`lumenairy/io/`, `lumenairy/optimize/`)

Findings I1–I8 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §7
(partition report `IO-OPTIMIZE.md`).  Branch `audit-fixes-2026-09`; no git
write commands were run.  Every python invocation used
`OPENBLAS_NUM_THREADS=1`.

## 1. Summary

| # | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| **I1** (P0) CODE V `DIM` units | **fixed** | `io/prescriptions_code_v.py:32-73,143-200,478-600` | `tests/unit/test_audit2609_a10_codev.py::test_i1_*` (9), `validation/io/test_io.py::t_codev_seq_dim_tokens` | the unit definitions (1 mm = 1e-3 m, 1 in = 0.0254 m exactly) + `system_abcd_prescription` for the end-to-end scale | `DIM M` R1 62.75 m → **0.06275 m**; EFL 72.2154 m → **0.0722154 m**; `DIM C` 62.75 m → 0.6275 m; `DIM I` 62.75 m → 1.59385 m; unknown token silent → warns |
| I1 writer default | **fixed** | `io/prescriptions_code_v.py:152-200,246-260` | `…_codev.py::test_i1_writer_emits_codev_millimetres`, `…::test_i1_writer_load_roundtrip_is_exact` | byte inspection of the emitted file + round-trip identity | `RDY 0.05000000` (metres, ×1000 wrong for CODE V) → **`RDY 50.00000000`**; round-trip max\|ΔR\| = 0.0 both before and after |
| I1 migration marker | **fixed** | `io/prescriptions_code_v.py:51-61,470-520` | `…_codev.py::test_i1_legacy_lumenairy_file_is_detected_and_read_as_metres` | banner+marker pairing; forced `dim_units='M'` reading | legacy file: values preserved (0.025 m) **with a loud warning**; `dim_units='M'` forces 2.5e-5 m |
| **I2** Zemax powered-surface window | **fixed** | `io/prescriptions_zemax.py:120-172,745-760,800-830` | `tests/unit/test_audit2609_a10_io.py::test_i2_*` (3) | the file's own surface list | PARAXIAL+STOP: `elements surf_nums [2,3]`, `stop_index None`, 0 warnings → **surface 1 present, `stop_index=0`, loud unsupported-SURFTYPE warning**; AC254 doublet gains no warning |
| **I3** CODE V `REFL`/`K`/`A…J`/`radius` | **fixed** | `io/prescriptions_code_v.py:63-73,690-780,830-900` | `…_codev.py::test_i3_*` (5) | closed-form: lens-unit→SI rule `a_p = a_file/L^(p-1)`; `has_mirrors`; `validate_prescription` | conic `0.0` → **−1.0**; asph `None` → **{4: 123.4, 6: −5.0e4}** (exact); mirror air→air → **`element_type='mirror'`, `has_mirrors` True**; `radius=None` (unusable) → **inf** |
| **I4** exporters drop `radius_y` | **fixed (warn)** | `io/prescriptions_zemax.py:2063-2110` (helper), `:1896`, `:2330`, `:2470`; `io/prescriptions_code_v.py:196` | `…_io.py::test_i4_*` (3), `…_codev.py::test_i4_codev_writer_warns_on_anamorphic_surface` | presence/absence of the diagnostic on a `make_cylindrical` export | `radius_y preserved=False, warnings=[]` → **`UserWarning` per anamorphic surface** on all three writers; symmetric surface still silent |
| **I5** codegen injection | **fixed** | `io/codegen.py:42-141,~700,~760,~880,~970` | `…_io.py::test_i5_*` (3, 2 payloads) | AST of the emitted line (one `Assign`, string-constant key) + `exec` of the registry block under `redirect_stdout` | emitted line parsed as **2 statements** and the payload executed → **1 `Assign`, no output from `exec`** |
| **I6** Zarr metadata codec | **fixed** | `io/storage.py:1353-1410,1490-1500,1545-1620` | `…_io.py::test_i6_zarr_append_plane_metadata_matches_hdf5` | the module's own 19-type probe set; a 4096-element ndarray | **13/19 → 18/19** faithful (matching HDF5); ndarray no longer stringified/truncated |
| **I6** `LA1509-C` catalogue radius | **fixed** | `io/prescriptions_builders.py:450-530` | `tests/unit/test_audit2609_a10_catalog.py` (9) | independent reduced-slope paraxial trace, cross-checked against `system_abcd` (agree < 1e-12) | R1 103.29 mm → **51.5 mm**; EFL@587.6 nm 199.865 mm → **99.652 mm** (nominal 100.0) |
| **I6** other catalogue rows | **partially fixed** (warns, data not corrected) | `io/prescriptions_builders.py:487-560` | `…_catalog.py::test_i6_catalog_row_efl_ledger` | same independent trace vs the part number | AC254-050-C −10.9 %, AC254-100-C −16.8 %, AC254-200-C −31.3 % — **now warn on every call**; needs vendor data (see §6) |
| **I7** `scale_prescription` coverage | **fixed** | `io/prescriptions_transforms.py:29-81,170-300` | `…_io.py::test_i7_scale_prescription_*` (2) | exact products at `s = 0.25` | `r_max` 0.0075 → **0.001875**; `q_bfs_coeffs` unscaled → **×s**; BFL 0.084 → **0.021**; DOE period 2e-6 → **5e-7**, `gap_before` 0.01 → **0.0025**; unknown length-like key now warns |
| **I7** codegen `inf`/`nan`/`-inf` | **fixed** | `io/codegen.py:110-141,~730,~990` | `…_io.py::test_i7_codegen_emits_parseable_inf_nan_and_keeps_the_sign` | executing the emitted block | `NameError: name 'inf' is not defined`, and a `-inf` radius written as `float('inf')` → **block executes; radius comes back `-inf`** |
| **I7** `normalize_prescription`→codegen | **fixed** | `io/codegen.py:458,499,583,590,429`; `io/prescriptions_transforms.py:300-320` | `…_io.py::test_i7_normalize_prescription_output_is_codegen_shaped` | the call itself | `KeyError: 'element_type'` → **script generated**, and `q['elements'] == q['surfaces']` still True |
| **I7** Zemax `MNUM`/`MCON` | **fixed** | `io/prescriptions_zemax.py:568-572,665-684,1327-1360` | `…_io.py::test_i7_multiconfig_*` (2) | the file's own records | no key, 0 warnings → **`configurations` dict + warning naming the operands**; single-config file → `None`, silent |
| **I7** HDF5 gzip default | **fixed (default changed)** | `io/storage.py:440-490,530-540,632-640,765-775,985-995` | `…_io.py::test_i7_auto_compression_*`, `…::test_i7_complex_field_writes_uncompressed_by_default`, `…::test_i7_append_plane_chunk_is_about_one_mib` | HDF5 filter pipeline + chunk shape read back from the file (no timing assertions) | gzip-4 vs None on 1024² c128, medians of 7 interleaved: **0.4274 s vs 0.0074 s write (×57.9), 0.0867 s vs 0.0122 s read (×7.1), 15.12 vs 16.01 MiB (−5.6 %)**; chunk 16 MiB → **1 MiB** |
| **I7** 31-plane focus scan | **fixed** | `optimize/context.py:259-292`; `optimize/driver.py:623-630,915-950`; `optimize/merit_terms.py` (5 flags + 2 forwards); `optimize/wrapper_merits.py` (3 forwards) | `tests/unit/test_audit2609_a10_optimize.py::test_i7_focus_scan_*` (3) | counting `through_focus_scan` calls/slices (a decision, not a timing) | opd-map merit: **3 scan calls / 93 slices → 0** (wave leg 32 → 1 propagation per eval, ×32); `StrehlMerit` unchanged at 31/call |
| **I7** multi-objective NaN front | **fixed** | `optimize/multi_objective.py:393-420` | `…_optimize.py::test_i7_infeasible_*` (2) | `np.asarray(None, float64)` measured `array(nan) ndim=0`; pymoo contract via a `sys.modules` stub | 0-d NaN array returned as the Pareto front (or `IndexError` with `progress`) → **`ValueError` naming the fix knobs** |
| **I7** CODE V/Quadoa `elements` | **fixed** | `io/prescriptions_code_v.py:790-900`; `io/prescriptions_quadoa.py:385-400`; `io/prescriptions_transforms.py:370-395` | `…_io.py::test_i7_quadoa_loader_emits_elements_and_all_thicknesses`, `…::test_i7_split_at_mirrors_warns_on_the_schema_free_fallback`, `…_codev.py::test_i3_refl_and_rmd_refl_import_as_mirrors` | key-set diff vs `load_zemax_zmx` | `.seq` 7 keys / `.qos` no `elements` → **both emit `elements` + `all_thicknesses`**; the silent fallback in `split_prescription_at_mirrors` now warns |
| **I7** unknown-glass warnings | **fixed** | `io/prescriptions_code_v.py:830-850`; `io/prescriptions_quadoa.py:412-428` | `…_codev.py::test_i7_codev_loader_warns_about_unknown_glasses` | `GLASS_REGISTRY` membership | silent (failure surfaced later inside a propagation) → **warns, naming the glasses** |
| **I7** edge-thickness constraint | **added** | `optimize/merit_terms.py:1562-1740`; `optimize/{__init__,core}.py` | `…_optimize.py::test_i7_edge_thickness_*` (4 + 4 params) | independently written closed-form spherical sag | did not exist → `edge_thickness` matches the oracle to **\|Δ\| ≤ 5.2e-18 m** on biconvex / concave-first meniscus / plano-convex / plate |
| **I8** `.zmx` `NAME`/`COMM` injection | **fixed** | `io/prescriptions_zemax.py:2063-2087,2065,2331` | `…_io.py::test_i8_zmx_name_newline_cannot_inject_records` | counting `SURF` records in the written file + reloading it | an extra `SURF 99` record → **4 records (object + 2 + image), reloads as 2 surfaces** |
| **I8** UTF-16-BE `.zmx` | **fixed** | `io/prescriptions_zemax.py:508-522` | `…_io.py::test_i8_utf16be_zmx_is_read_not_misdiagnosed` | a real BE-encoded file | `OSError: does not appear to be a Zemax .zmx lens file` → **loads, R1 = 50 mm** |
| **I8** TOROIDAL/BICONICX → `radius_y` | **deferred** (diagnostic improved) | `io/prescriptions_zemax.py:944-960` | — | — | still imports as base conic with a loud warning; the warning now names the hand-entry route.  See §6 |
| **I8** `export_quadoa_qos` stop invention | **fixed** | `io/prescriptions_quadoa.py:171-183,215,235` | `…_io.py::test_i8_quadoa_writer_does_not_invent_a_stop` | round-trip of a prescription with no declared stop | `stop_index = 0` invented on every round trip → **`None`**; a declared stop still round-trips |
| **I8** `create_zoom_configs` | **fixed** | `optimize/multiconfig.py:148-215` | `…_optimize.py::test_i8_zoom_configs_reject_a_mis_sized_spacing_vector` | slot count | silent truncation + glass thicknesses overwritten → **`ValueError`**, plus a `(slot, value)` pair form |
| **I8** `method='newton'` bounds | **fixed** | `optimize/driver.py:1453-1470` | `…_optimize.py::test_i8_newton_*` (2) | presence/absence of the warning | silent → **warns**, matching the generic and `lm` branches; no bounds → still silent |
| **I8** `x0` bounds check | **fixed** | `optimize/multi_objective.py:255-275` | `…_optimize.py::test_i8_x0_outside_bounds_warns` | source ordering (pymoo absent) | never checked → **warns** |
| **I8** ndarray aperture cache key | **fixed** | `optimize/wrapper_merits.py:46,133-160` | `…_optimize.py::test_i8_aperture_cache_key_is_a_content_digest` | key equality/inequality on arrays differing in one element | 64-bit `hash(bytes)` → **128-bit `blake2b` digest** |
| perf note 6: zarr drops `compression` | **fixed (warn)** | `io/storage.py:1857-1875` | — | — | silently dropped → **warns** |

**Confirmed to fail on the pre-fix behaviour.**  Beyond quoting the audit's own
pre-fix measurements in every test docstring, seven properties were re-checked
by reverting the fix **in process** and re-asserting
(`scratchpad/prefix_check.py`): I1 DIM scale (62.75 m vs 0.06275 m), I1 writer
numbers, I3 `K`+`REFL` (conic 0.0 / `has_mirrors` False), I5 injection
(emitted line parses as 2 statements), I6 zarr metadata (5/5 probe types lost),
I7 focus-scan gate (3 scans / 93 slices → 0), I8 aperture digest (int → bytes).
**7/7 confirmed.**

## 2. Per finding

### I1 (P0) — CODE V `DIM` units

**Wrong.** `_unit_to_meters` used `{'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}` with
`units='M'` as the default, and `if unit_tok in ('M','MM','IN')` meant any other
token was ignored and left the default in force.  The writer mirrored it
(`scale = {'M': 1.0, ...}`).  CODE V's `DIM` takes exactly `M` (millimetres),
`C` (centimetres), `I` (inches); it has no metre unit.  So every genuine `.seq`
imported ×1000 too large in silence, and every file the library wrote was ×1000
too large for CODE V.

**Changed.** A single `_CV_DIM_TO_METERS` table (`{'M': 1e-3, 'C': 1e-2, 'I':
0.0254}` plus tolerated read-side aliases `MM`/`CM`/`IN`/`INCH`), an unknown-token
`UserWarning` modelled on the Zemax loader's S4-9 warning, a writer that
normalises aliases to the CODE V token it means and scales by `1/L`, and a
`dim_units=` override on the loader.

**Migration.** The writer stamps `! LUMENAIRY-SEQ-FORMAT 2` next to its existing
generator banner.  The loader treats *banner present + marker absent* as "written
by lumenairy before v5.46, values are SI metres", reads it on those terms (so the
numbers are unchanged) and warns.  This is the only way to tell such a file from
a genuine CODE V one, and it keeps every file already on disk round-tripping.

**Verified.** `repro/IO-OPTIMIZE/p2_codev.py` before → after as tabulated above;
`export→load` identity max |ΔR| = 0.0; the two repo fixtures that pinned the
wrong convention were corrected (see §5 for the exact edits).

**Residual risk.** A user who hand-edited an old lumenairy `.seq` into real CODE V
units but kept the banner will now be read as legacy (metres).  The warning names
`dim_units='M'` as the override, and re-exporting stamps the marker.

### I2 — Zemax powered air-to-air surfaces

**Wrong.** The window auto-detect's `active` predicate admitted glass, mirrors and
DGRATINGs only, so a `PARAXIAL` / `ABCD` / phase surface was deleted *before* the
unsupported-SURFTYPE branch could warn — taking its STOP flag and DIAM with it.

**Changed.** `_raw_surface_is_air_powered` (a silent predicate, exactly like
`_raw_surface_is_dgrating`) admits the SURFTYPEs that act without glass when they
carry a non-empty `PARM` table or a curvature.  Separately, *any* optical surface
the final window excludes that carries a non-zero `CURV`, a non-empty `PARM` table
or the STOP flag is named once in a warning — belt and braces for the SURFTYPEs
the predicate cannot know.

**Residual risk.** The PARAXIAL surface still imports as a *flat* air-to-air
surface (the prescription schema has no ideal-lens element), so its 100 mm power
is not modelled — but the loud warning now says so, where before the surface
vanished silently.  A PARAXIAL-**only** file still raises (now "Need at least 2
surfaces", where before it raised "No glass/mirror/diffractive surfaces found").

### I3 — CODE V `REFL` / `K` / `A…J` / `radius=None`

**Changed.** `REFL` and `RMD REFL` set a mirror flag that produces an
`element_type='mirror'` entry in a new `elements` list; `K` joins `CON` as a conic
keyword (the writer now emits `K`, CODE V's own); `A`…`J` are parsed with the
lens-unit rescale and emitted back as an `ASP` block; the `radius` default is
applied where the `.get` default was dead.  The refractive-only `surfaces` list
folds the mirror legs flat, the same collapse `load_zemax_zmx` performs.

**Verified.** Exact coefficient identities (`A 1.234E-07` → 123.4 m^-3,
`B -5.0E-11` → −5.0e4 m^-5); mirror + conic + asphere survive a
load→export→load round trip to rel 1e-8 (the writer's `%.10E` / `%.8f` field
widths).

**Residual risk.** `XDE/YDE/ZDE/ADE/BDE/CDE` (fold decenters/tilts) are still
unparsed and still warn (CV-1, unchanged).  A CODE V mirror inside glass keeps
the incoming medium; a `GLA` on the mirror row overrides it.

### I4 — anamorphic keys dropped by the writers

Taken as the audit's stated minimum ("warn as loudly as `_warn_dropped_qtype`"),
applied to all three writers (`export_zemax_zmx` both paths,
`export_zemax_lens_data`, `export_codev_seq`).  Emitting `BICONICX` / `YTO`+`CUY`
is deferred — see §6.

### I5 — codegen injection (security)

**Wrong.** The raw `GLAS` token, the labels derived from it (`_lens_group_name`)
and the system name went into code positions unescaped.

**Changed.** (a) every interpolated string is `repr()`-ed or collapsed through
`_comment_text` (which also neutralises a `"""` that would close the module
docstring); (b) `_validate_codegen_token` reduces glass / system names to
`[A-Za-z0-9_\-.+]+` at the boundary and warns when anything is stripped;
(c) `_py()` renders every numeric literal.

**Verified.** For both of the audit's payloads: the emitted script still parses,
every `GLASS_REGISTRY` line is exactly one `ast.Assign` with a string-constant
subscript key, and executing the registry block against a stub registry under
`redirect_stdout` produces **empty** output.  A `"""`-bearing prescription name
produces no `Import` of `os` and no `os.system` call node.

**Residual risk.** Residual alphanumerics of a payload can still appear *inside*
the quoted key (e.g. `'Xprint0x50574e4544'`) — inert, and the warning tells the
user the token was doctored.  The generated script still imports `lumenairy` and
runs the user's own design; this fix is about the *file* not being able to inject
code, not about sandboxing.

### I6 — Zarr metadata + the catalogue

Both halves of the audit row.  The zarr writer/readers now use the canonical
codec (`_zarr_write_meta_attrs` / `_zarr_read_attrs`, twins of the HDF5 pair);
`_zarr_write_sim_metadata` was refactored onto the same helper.  The catalogue
half corrected `LA1509-C` and added an EFL ledger + per-call warning; the three
`AC254-*-C` rows are wrong by 11–31 % and now say so (§6).

**Residual risk.** The `np.float32 → float` miss is inherent to the JSON lowering
and is shared with HDF5 (18/19 on both).  Pre-existing zarr stores carry no
metadata blob and read back exactly as before.

### I7 / I8

Each item is covered in the summary table with its before/after numbers.  Two
deserve a note:

* **HDF5 compression default.**  This is a **deliberate default change**, licensed
  by the finding ("the default is wrong for complex data").  `'auto'` keeps gzip-4
  for every non-complex array, so only complex fields move.  The measurement is
  medians of 7 interleaved runs specifically because this box is shared — a
  non-interleaved run of the audit's own probe on the same machine produced an
  *inverted* ordering (gzip 20 s vs None 55 s at 4096²) purely from I/O
  contention, which is exactly the trap `TESTING_STANDARDS` S1 describes.  No
  timing is asserted in any test: the tests read the HDF5 filter pipeline and the
  chunk shape back out of the written file.
* **Focus-scan gating.**  `needs_focus_scan` defaults to `True`, so a
  user-written merit class that predates the flag keeps its scan and its results
  are bit-identical.  Only the library's own opd-map merits opt out.

## 3. Files touched

Source (all within the WP's ownership):

```
lumenairy/io/prescriptions_code_v.py      lumenairy/optimize/context.py
lumenairy/io/prescriptions_zemax.py       lumenairy/optimize/driver.py
lumenairy/io/prescriptions_quadoa.py      lumenairy/optimize/merit_terms.py
lumenairy/io/prescriptions_builders.py    lumenairy/optimize/wrapper_merits.py
lumenairy/io/prescriptions_transforms.py  lumenairy/optimize/multi_objective.py
lumenairy/io/codegen.py                   lumenairy/optimize/multiconfig.py
lumenairy/io/storage.py                   lumenairy/optimize/__init__.py
                                          lumenairy/optimize/core.py
```

Fixtures / tests corrected (each encoded a wrong convention or value — the WP
explicitly assigns these):

```
tests/unit/test_audit_misc.py         D.2 CODE V .seq fixture: wrote SI metres
                                      under `DIM M`; now genuine CODE V mm, with
                                      the radii/thickness/aperture scale asserted
validation/io/test_io.py              t_codev_seq_units_mm required the writer to
                                      emit `DIM MM` (not a CODE V token); now
                                      asserts `DIM M` + millimetre numbers.  New
                                      t_codev_seq_dim_tokens covers M/C/I + the
                                      unknown-token warning
tests/unit/test_niche_d4_dgrating.py  the exact key-set pin on load_zemax_zmx's
                                      return now includes `configurations`
validation/real_lens_opd/lens_cases.py            LA1509-C described as f=200 mm
validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx  CURV 0.0096814793 ->
                                                           0.0194174757
validation/real_lens_opd/zemax_prescriptions/LA1509_C.txt  regenerated
validation/real_lens_opd/zemax_prescriptions/INDEX.md      LA1509_C row only
```

New test files:

```
tests/unit/test_audit2609_a10_codev.py      19 tests  (I1, I3, I4, I7)
tests/unit/test_audit2609_a10_io.py         24 tests  (I2, I4, I5, I6, I7, I8)
tests/unit/test_audit2609_a10_optimize.py   17 tests  (I7, I8)
tests/unit/test_audit2609_a10_catalog.py     9 tests  (I6 data half)
```

`lumenairy/__init__.py` was **not** modified (an early edit adding the two new
`optimize` names was reverted; the file matches `HEAD`).  See §5.

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a10_{codev,io,optimize,catalog}.py` | **69 passed** | 2.4 s |
| `pytest tests/unit/test_audit_io.py test_audit_optimize.py test_audit_s4_9_io_silent_fallback.py test_g08_s4_19_io_hygiene.py test_v5_1_0_agent_f_split.py test_v5_4_6_io_ui_delegated.py test_combine_prescriptions.py test_niche_d4_dgrating.py test_niche_c1_consolidation.py` | 341 passed, 1 skipped, **4 failed** (2 mine, fixed below; 2 not mine) | 465 s |
| re-run after fixing the two schema pins: `pytest tests/unit/test_niche_d4_dgrating.py test_v5_1_0_agent_f_split.py` | **146 passed** | 193 s |
| `pytest tests/unit/test_audit_w5_zemax.py test_audit_w6_io_zemax.py test_audit_w5_optimize.py test_audit_w6_optimize.py test_optimize_merit_terms.py test_c1_s4_19_storage_metadata_contract.py test_g08_s4_18_optimizer_hygiene.py` | **79 passed** | 5.1 s |
| `pytest tests/unit/test_audit_s4_2_… s4_5_… s4_6_… s4_7_… test_v5_6_raw_parameterization.py test_v5_3_multi_field_merit_jit.py test_audit_v5_24_2_b3_merit_scale.py test_v4_16_0_agent_b_multiprocess_storage.py` | **100 passed, 8 skipped** (PySide6 / complex256 absent) | 17 s |
| `pytest tests/unit/test_audit_misc.py -k "codev or quadoa or storage or zemax or d2 or d3"` | **9 passed**, 219 deselected | 101 s |
| `pytest tests/unit/test_folded_design_guard.py test_v5_21_2_subsystem_audits.py test_plot_lens_layout_ray_overlay.py test_g08_s4_15_cache_hygiene.py test_niche_audit_r_guards_and_merits.py test_v4_15_agent_f.py` | **129 passed, 3 skipped** (astropy / jax-x64) | 32 s |
| `python validation/run_all.py io` | **32/32 passed** | 4.1 s |
| `python validation/run_all.py optimize` | **23/23 passed** | 7.3 s |

**Two existing tests pinned the pre-fix schema and were updated in place**
(both are about modules this WP owns):

1. `test_v5_1_0_agent_f_split.py::test_make_singlet_round_trips_through_transforms`
   asserts `nrx['elements'] == nrx['surfaces']`.  My first implementation of the
   `normalize_prescription` `element_type` stamp used shallow copies and broke
   that identity.  **I changed the implementation, not the test**: the stamp is
   applied in place on the shared dicts, so the documented identity (and the
   aliasing it rests on) is preserved and both views carry the discriminator.
   The test is unmodified.
2. `test_niche_d4_dgrating.py::TestDgratingImport::test_no_dgrating_is_additive_only`
   pins the exact key set `load_zemax_zmx` returns ("must only GAIN
   'diffractives'").  The new `configurations` key (I7) is additive in exactly
   the same way, so the pin was widened to include it, with a comment saying
   why, plus a new assertion that a single-config file gets `configurations is
   None`.

**Pre-existing failures found, in my judgement unrelated to WP-A10** (both in
`tests/unit/test_niche_c1_consolidation.py`, both in other WPs' territory —
confirmed by locating the symbols they read):

* `test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes` — the
  whitelist gained `on_focus_containment`, which is defined in
  `lumenairy/propagators/carrier.py:2637` (the CARRIER work package).  Nothing in
  `io/` or `optimize/` touches that whitelist.
* `test_breaking_the_tilted_path_moves_the_skew_spot_off_the_oracle[no tilt ramp]`
  — the `surface_frame` tilt-ramp path (`elements/_lens_real.py`, finding F-O2 of
  ORCHESTRATOR.md).

Also observed while benchmarking, and **not** investigated further because it is
outside this WP: `save_field_h5(..., compression='lzf')` on a 1024² complex128
array did not complete within 400 s on this machine, while `gzip` took 0.43 s and
`None` 0.007 s.  `lzf` is not a default anywhere and no test exercises it; it is
worth a look (h5py filter or plugin issue, most likely not library code).

## 5. Requested changes outside my ownership

1. **`lumenairy/__init__.py` — re-export the two new `optimize` names.**  COMMON.md
   forbids me editing this file, so the new public API is currently reachable only
   as `lumenairy.optimize.MinEdgeThicknessMerit` / `lumenairy.optimize.edge_thickness`.
   Requested change, mirroring the adjacent `MinThicknessMerit` lines:

   * in the `from .optimize import (...)` block (next to `MinThicknessMerit,`), add
     `MinEdgeThicknessMerit,` and `edge_thickness,`;
   * in `__all__` (next to `'MinThicknessMerit',`), add `'MinEdgeThicknessMerit',`
     and `'edge_thickness',`.

   Both names already exist in `lumenairy/optimize/__init__.py.__all__` and in
   `lumenairy/optimize/core.py`, so the import will resolve.  My tests import from
   `lumenairy.optimize`, so they pass either way.

2. **`lumenairy/analysis/plotting.py:2012` — a stale measured ratio in a comment.**
   The `|image_z| / max(track, aperture, |efl|)` census comment reads
   `LA1509-C 0.9883`.  With the corrected 100 mm radius that lens's ratio changes
   (the test that consumes the bar, `test_plot_lens_layout_ray_overlay.py`, still
   passes — the bar is 10 and every entry is ≤ 1).  Suggested: re-run the census
   and update the one number, or add "(LA1509-C re-measured after the I6
   catalogue correction)".

3. **`CHANGELOG.md` / `Migration-Guide.md`** — the orchestrator assembles these
   from `WP-A10_CHANGELOG.md`.  The two migration notes that need to reach the
   user-facing guide are the CODE V `DIM` convention (with the
   `LUMENAIRY-SEQ-FORMAT 2` marker and the `dim_units=` escape hatch) and the HDF5
   `compression='auto'` default.

4. **`README.md:3177-3179`** — "CODE V `.seq` import / export … (units M/MM/IN …)"
   should read "units M (mm) / C (cm) / I (inch), with MM/CM/IN tolerated as
   aliases on read".

## 6. Deferred items

1. **`AC254-050-C` / `AC254-100-C` / `AC254-200-C` catalogue rows** (I6 remainder).
   Measured paraxial EFLs 44.560 / 83.171 / 137.395 mm against nominal 50 / 100 /
   200 mm — 10.9 %, 16.8 % and 31.3 % off.  The header comment's claim "Surface
   data from Thorlabs Zemax files" is therefore false for these three.  I did not
   invent replacement radii: the audit brief supplied ground truth for LA1509 only,
   and fabricating vendor data is precisely the defect class being fixed.
   **Design:** take R1/R2/R3, tc1/tc2 and the glass pair from the Thorlabs
   `AC254-0xx-C` Zemax files (the `-C` series uses different glasses from the `-A`
   series, which is the likely origin of the error — the current rows pair `-A`-era
   radii with `-C`-era glasses), then re-run
   `validation/real_lens_opd/export_all_zemax.py` to regenerate those three
   fixtures and `INDEX.md`.  **Effort:** ~1 h once the vendor data is to hand; the
   ledger test then flips from "known-bad" to the < 3 % arm automatically (it will
   FAIL until the ledger is updated, which is the intent).
   **Mitigation in place:** `thorlabs_lens()` warns loudly on every call.

2. **Zemax `TOROIDAL` / `BICONICX` → `radius_y` / `conic_y`** (I8).  The library has
   full biconic support, and the PARM slots map onto it — but whether a `.zmx`
   `BICONICX` PARM carries an X *radius* or an X *curvature* (and which of
   base-`CURV` / PARM is the X vs Y profile) is not verifiable offline, and a wrong
   mapping silently produces a wrong surface, which is worse than the current
   honest "imported as base conic" warning.  **Design:** confirm the two
   conventions against an OpticStudio-written `BICONICX` file (one surface, X and
   Y radii deliberately different, e.g. 50 / 80 mm), then map base `CURV`/`CONI` →
   the Y profile and `PARM 1`/`PARM 2` → `radius`/`conic`, add the matching
   `export_zemax_zmx` emission path (which also closes I4 properly), and pin the
   round trip on that fixture.  **Effort:** ~2 h with a reference file; 0 without
   one.  **Mitigation in place:** the unsupported-SURFTYPE warning now names
   `make_biconic` / `radius_y` / `conic_y` as the hand-entry route.

3. **Coarse-to-fine / cached through-focus scan** (I7 options b and c).  Option (a)
   — gate the scan on whether any merit reads it — is implemented and is the whole
   win for the opd-map merit families.  For a `StrehlMerit` run the scan is still
   31 planes per evaluation.  **Design:** add `z_scan_mode='full' | 'coarse'` to
   `design_optimize`; `'coarse'` runs 7 planes, fits a parabola to the Strehl peak
   and refines with 5 planes around it (≈2.6× cheaper for the same best-focus
   resolution), and caches the scan across an FD stencil by keying it on the
   stencil centre.  It must NOT be the default (it changes `z_best` resolution),
   and it needs its own accuracy oracle: `|z_best(coarse) − z_best(full)|` below
   the Rayleigh range on a fixture ladder.  **Effort:** ~3 h including the oracle.

4. **Parser consolidation** (audit "Alternative algorithms").  The `DIM` bug (P0),
   the missing `REFL` (P1) and the missing unknown-glass warning (P2) all existed
   in one loader and not the others because there are four hand-rolled
   `if keyword ==` ladders with independently-evolved unit handling, warning policy
   and output schema.  This pass narrowed the divergence (all four loaders now warn
   about unknown glasses; three now emit `elements`/`all_thicknesses`) but did not
   remove it.  **Design:** one `Record(keyword, arity, units, handler)` table per
   format plus a single `PrescriptionSchema` builder.  **Effort:** 1–2 days;
   properly a separate work package.

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A10_CHANGELOG.md`

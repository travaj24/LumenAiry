# WP-A18 — Documentation: README / Migration-Guide / CONVENTIONS / ROADMAP + a living subsystem document

Branch `audit-fixes-2026-09`.  No source file, no test file and no git write command was
touched.  Files changed: `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`,
`ROADMAP.md`, new `docs/subsystems/real_lens.md`, plus this report and
`fixes/WP-A18_CHANGELOG.md`.

---

## 1. Summary

| deliverable | status | evidence |
|---|---|---|
| **D1** non-resolving identifiers resolved / removed; resolver run with a clean denominator | done | 4 unresolved -> **0** on a 592-token API-claiming denominator (§2) |
| **D2** every convention sentence the WPs requested, in `CONVENTIONS.md`, each with a "Changelog of this file" entry | done | power (L13), tilt as IMPLEMENTED (L2/L19), Richards--Wolf pupil (K10), §11 `fff_nv` + `formulation='auto'` (H3/G6), §7.1 confirmed (G4/G13), §10 + new §10.1 (H5/V6) (§3) |
| **D3** one migration note per changed default / convention, grouped by area, taken in substance from the WP changelogs | done | new `Migration-Guide.md` section "5.46.0 -- adversarial audit remediation", 11 area groups, 648 lines (§4) |
| **D4** `docs/subsystems/real_lens.md` < ~400 lines | done | 343 lines; both code blocks execute (§5) |
| **D5** README landing section + changed-default snippets + the units text at ~3177 | done | Start-here table; CODE V units per WP-A10 §5.4; `LGAberrationMerit` scale note; 10 of 14 fenced blocks execute, the 3 failures pre-existing (§6) |
| **V** doc walkers / tests that read the four files | run | 216 passed, 2 skipped, 1 failed — the failure is another WP's uncommitted `.py` edit (§7) |

---

## 2. D1 — the identifier sweep

### 2.1 Method

A resolver (kept at
`<scratch>/wpa18/resolve_ids.py`; I own no `.py` in the repo so it is not committed)
extracts every backticked token from the four files, normalises RST-style ``double
backticks``, keeps the identifier-shaped ones, classifies each, and resolves the
API-claiming subset against the **currently importable package**:

* dotted paths rooted at `lumenairy` / `la`, walking modules then attributes;
* `Foo.bar` chains against every module that exposes `Foo`;
* bare names against `lumenairy`, then every submodule's `dir()`, then module leaf names;
* a **static AST index** of `lumenairy/**/*.py` for module attributes, class members
  (including `self.x = ...`) and nested `def` / `class` definitions — this is what makes
  `lumenairy/ui/` resolvable at all, since PySide6 is not installed on this box, and what
  makes `Surface.world_origin`, `MultiFieldMerit.field_angles`, `_RestoreDtype` and
  `_merit_jac_auto` resolve honestly rather than being counted as breakage.

Exclusions are in two layers so the denominator is auditable:

1. **mechanical** — exception / builtin names, kwarg and parameter names (taken from the
   signatures of every callable in the package), test citations, third-party and stdlib
   names, file names, enum-like option string values, and file-format record tokens
   (`DIM`, `RDY`, `THI`, …);
2. **hand-triaged** — 51 tokens inspected at their citation site, each with a written
   reason in the script (removed / renamed API that a migration guide must be able to
   name; local expressions inside an illustrative snippet; test-corpus names; ROADMAP
   names for things that do not exist yet; other projects' API named as a comparison;
   placeholders in a naming rule; designer element labels; dict keys).

### 2.2 Counts, before and after (same script, same rules)

"Before" is `git show HEAD:<file>` for all four files; "after" is the working tree.

| | occurrences | distinct identifier-shaped | denominator (mechanical exclusions only) | resolve | unresolved |
|---|---:|---:|---:|---:|---:|
| **before** | 1 673 | 923 | 576 | 525 | **51** |
| **after** | 1 952 | 1 055 | 644 | 593 | **51** |

| | denominator (fully triaged) | resolve | unresolved |
|---|---:|---:|---:|
| **before** | 528 | 524 | **4** |
| **after** | **592** | **592** | **0** |

Unresolved occurrences per file, fully triaged: before README 5 / ROADMAP 0 /
Migration-Guide 0 / CONVENTIONS 0; after **0 / 0 / 0 / 0**.

The two tables say different things and both are honest.  The *mechanical* row is
unchanged at 51 because my edits removed four stale references and the new ROADMAP
section deliberately adds four names that do not exist yet (`LensGeometry`,
`LensNumerics`, `LensResources` — the audit's own proposed config objects) plus one
quoted warning text.  The *triaged* row is the deliverable: after this pass every
backticked identifier in the four files either resolves against the current package or
sits on an explicit, reasoned exclusion list.

For scale: the audit sampled 60 tokens with `random.seed(0)` and found 13 resolving /
47 not (78 %), while saying "*that headline overstates the problem*… I did not build a
clean denominator".  The 78 % is inflated by exception names, parameter names and test
citations, and by resolving only against `lumenairy` and its nine subpackages rather than
against deep modules.

### 2.3 Disposition of the twelve identifiers the audit named

| identifier | disposition |
|---|---|
| `_decompose_prescription` | **resolves** (`lumenairy.io.codegen`, private).  README now cites it as `io.codegen._decompose_prescription`. |
| `focus_fixed_sampling` | **prysm's** API, named as a comparison.  The README line now says so explicitly (with POPPY's `apply_image_plane_fftmft`). |
| `_detect_backend` | **resolves** (`lumenairy.io.storage`).  ROADMAP now cites `io.storage._detect_backend`. |
| `return_kind` | real kwarg of `create_gaussian_schell_source`.  No change. |
| `opl_fn` | real kwarg of `propagate_huygens_fresnel_with_opl_callable`.  No change. |
| `image_centres` | real kwarg of `combine_patch_fields`.  No change. |
| `world_origin`, `world_R` | real `raytrace.Surface` fields.  No change (the README sentence already introduces them as `Surface` fields). |
| `rcwa_1d` | removed in 4.4.  The first citation already says "removed"; the second now says so too and names `rcwa_efficiency_1d`. |
| `_PROPAGATE_SYSTEM_JAX_CACHE` | **resolves** (`lumenairy.propagators.system`). |
| `row_reset` | was a dead local name (`last_v_star`) in a v4.14 release note.  Rewritten as the `maslov_tracking='row_reset'` option it describes; `last_v_star` removed. |
| `fiber_mode` | a source-KIND string, not a symbol.  Now quoted as `'fiber_mode'`; the test-class citation beside it replaced by prose. |
| `_spawn_rng`, `_fd_grad_pure` | **resolve** (`propagators.hfpi`, `optimize.core`). |

Two further stale references found and fixed by the sweep:
`raytrace._intersect_surface` -> `raytrace.intersection._intersect_surface`, and
`lumenairy.asymptotic` -> `lumenairy.propagators.asymptotic` in the *current* feature
list (the v3.3 historical mention keeps the old name and states the v5.0 relocation).

---

## 3. D2 — `CONVENTIONS.md`

| requested by | change |
|---|---|
| WP-A2 §5.1 / VERIFY-A2 §5.5, finding **L13** | §7 gains a **Field normalisation and power** row and a bullet: `sum(abs(E)**2)*dx*dy` IS the power; an index step needs `T = (n2 cos theta_t)/(n1 cos theta_i) abs(t)**2`.  Measured single AIR->N-BK7 face 0.632344 -> 0.958057, against the closed form to 1.1e-16. |
| orchestrator ruling 2026-09-12, findings **L2 / L19** | The `Tilt axis convention` row claimed `tilt=(theta_x, theta_y, theta_z)`.  **Deleted** and replaced with the implemented convention, read from `_lens_real.py`'s `_sf_active` block (`_sf_thx = tilt[1]`, `_sf_thy = -tilt[0]`), `raytrace/surface.py::_field_frame_sag_and_grad` (`f += tx*xs + ty*ys`, `dfdx += tx`, `dfdy += ty`) and WP-A2_REPORT §2 (L2+L19): **`tilt = (t0, t1)` IS the sag ramp `t0*(x-dcx) + t1*(y-dcy)`, i.e. `theta_x = t1`, `theta_y = -t0`**.  The row names its five consumers and the cross-model pin.  A following bullet disambiguates it from the coordinate-break `tilt_*_deg` keys.  **No `new_tilt = (old_t1, -old_t0)` note anywhere** — nothing migrates. |
| WP-A5 §5.3, finding **K10** (+ K16) | §7 gains the **Richards--Wolf `pupil` indexing** row: the physical exit-pupil coordinate, not the projected ray direction; the two differ by a point inversion. |
| WP-A13 §5.2, finding **G13** | §7.1 **confirmed unchanged**: WP-A12's committed text already states "index 0 = `x`, index 1 = `y`" and that at `phi = 0` the x column is p/tm and the y column s/te, with `J[0,0] = -r_p` measured at 0/30/60 deg.  That is exactly WP-A13's request; re-stating it would have risked breaking `test_audit2609_a12_pmm1d.py::test_g4_the_1d_jones_is_the_lab_cartesian_basis_and_jxx_is_minus_rp`, which asserts both the presence of `J[0, 0] = -r_p` and the ABSENCE of the old sentence.  That test passes. |
| WP-A14 §5c, finding **H3**; WP-A13 **G6** | §11 records `rcwa_jones_2d(formulation='fff_nv')`'s validated-scope warning on a curved cell and `allow_nonseparable_nv`, and `pmm_jones_2d`'s `formulation='auto'` with the reflection-vs-transmission ordering caveat. |
| WP-A15a §5.11, findings **H5 / V6** | §10 records `threadpoolctl` as a HARD dependency (with the 140x / 400x measurements); new §10.1 documents `lumenairy.override(**knobs)`, the 17 registered knobs and the rule for registering a new one. |

Seven entries were added to the file's own "Changelog of this file" section, one per
change above, each naming the finding ID and (for the tilt row) what the old text claimed
and why it was wrong.

The WP-A15a suggestion to note the suite's real runtime went into `Migration-Guide.md`'s
packaging group rather than `CONVENTIONS.md`, which is an API-conventions document.

---

## 4. D3 — `Migration-Guide.md`

New section **"5.46.0 -- adversarial audit remediation (2026-09-11)"**, 648 lines, with
a preamble stating the two rules that apply throughout (broken *recommended* options were
fixed in place, so the call site does not change — only the answer; and the escape hatch
is named wherever the pre-fix behaviour is reproducible).  Groups and the notes each
carries:

* **Analytic lens** — `seidel_correction` (L1), `slant_correction` (L12) and its refusal
  with `seidel_correction` (L20), `fresnel=True`'s per-interface POWER factor (L13),
  `surface_frame=True` (L2/L19) **with the explicit statement that no stored `tilt`
  migrates**, `displaced` phase transport (L3/L7), `absorption` (L19), `stop_index`
  raising (L14), `screen_obliquity` (L4), the two stale caches (L5/L6).
* **Traced lens / caustic siblings** — the exit-vertex fix (S1), new `caustic='wave'`,
  `form_error` honoured (T5), vignetting in the answer, the exit-NA guard priced on the
  exit medium, T2/T3/T4/T12, `_reverse_prescription` (T6), `on_noncollimated`,
  segmentation.
* **Maslov / GBD / FGA / asymptotic** — `normalize_output='none'` scale (S4),
  `integration_method='auto'` (S2), the exit-vertex chart (S3), `stop_index`, the vector
  wrapper's joint normalisation, the three deprecated GBD reconcilers (S5), the JAX x64
  requirement (S7), the **Van Vleck weight** (Y2) with the two measured factors and
  `van_vleck_weight`, Y1, and the **dimensionless LG merit** (VERIFY-A4) with the
  `-4.79e+14` / `+0.9968` -> `[-3e-03, 1]` move.
* **Propagator kernels** — Richards--Wolf `E_z` sign (K16) and the `pupil` convention
  (K10), RS `kernel='auto'` (K9) with the bit-identity statement above the threshold,
  HFPI `normalisation='physical'` (K13/K18/V1), `rng=None` (K19), keyword-required
  `wavelength` (K12), `vector_projection=True` (K17), K20, `chunk_output` (K22), SAS (K3),
  `jv` (K2), K11/K21/K6.
* **Traced-carrier chain** — the containment guard and standoff (C1), `aggregate` dtype,
  complex64 preservation (C3), `carrier_referenced_fit_radius` (C2), and the
  **`CarrierField` mutation deprecation** with its v5.48 horizon and the
  `np.add(..., out=...)` replacement.
* **Analysis** — `wave_opd_2d` anchor (A1), the exact-sphere Strehl (A2), Shack--Hartmann
  half-factor + `reconstruction=` + **un-measured lenslets -> NaN** (A3),
  `object_distance=inf` + `field_max_rad` + **the infinite-conjugate field sign**
  (`field=(0,+1)` is an object above the axis at both conjugates; `L = -sin(Hx·field_max_rad)`,
  the negative of `ray_fan`'s `field_angle`) (A4), GS error scale and JAX dtype (A5).
* **Thin elements / materials** — glass rows (E1), out-of-range raise (E2), turbulence
  `sqrt(2)` with `r0*2**(-3/5)` (E3), GRIN `thin_form=True` (E5), `make_bsdf` strictness
  and the changed BSDF RNG stream (E6), lower-case `'air'` (E7), gray aperture.
* **Polarization / sources / algebra / memory** — coatings `polarization` strictness (Z1),
  Schell ensembles + `pad_sigma=0.0` (Z2), **`FreeSpace` default `'asm'`** with
  `method='auto'` as the way back and the new window-clipping report (Z3),
  `estimate_lens_memory(lens_model=)` (Z3), `from_prescription` shadowing (Z4).
* **Rigorous solvers** — `min_feature` default `1e-3·P` **and the snap warning** (G2),
  the JAX guard twin (G1), **2-D Li routing** (G5), `formulation='auto'` (G6) and
  `return_jones_transmission` (G13), the prepared-sweep fold with `symmetry=False` (G8),
  `max_pencil_dof` and the **joint redundancy advisory** (staggered), the **Wood symmetric
  bracket + `wl_eff`** (H2), `fff_nv` (H3), `guided_modes` (H1), G4, `pmm_2d_order_drift` (G7).
* **I/O / optimiser** — CODE V `DIM` marker and `dim_units=` (I1), HDF5
  `compression='auto'` (I7), LA1509-C (I6), I2/I3/I4/I7, `method='newton'` bounds (I8).
* **Designer UI / library** — the `object_distance` convention (U3), `is_mirror` /
  `semi_diameter` / `is_stop` export (U1/U2), the **`load_lens` back-fill** with its
  guard conditions and the once-per-entry `UserWarning`, U4.
* **Packaging / config / public API** — `threadpoolctl`, **`override()` and the knob
  registry** (with a runnable snippet), the 15 promoted names, **`rs_alias_free_distance`**,
  the shared exit-vertex transfer with the "replace every hand-written block" instruction
  and the grazing-policy change, `opd_fan_data`'s reference sphere and the warning that
  thresholds calibrated on the old fan numbers need re-checking, and the corrected suite
  contract.

Also: the stale "This guide covers v4.x. v5.0 will introduce its own migration section
when released" preamble was corrected, and the v5.2.0 `surface_frame` section gained a
superseded-in-part banner linking forward.

**VERIFY-A4's changelog file exists** (`fixes/VERIFY_WP-A4_CHANGELOG.md`, last written
10:12 today), so the Van Vleck weight and the dimensionless LG merit are documented from
it rather than listed as pending.

---

## 5. D4 — `docs/subsystems/real_lens.md`

343 lines.  Nine sections: the family at a glance (9 entry points + the two JAX twins and
their x64 requirement); the exit-vertex invariant with a runnable demonstration; the
analytic model (`surface_model`, the per-surface modifiers, the `tilt` key); the traced
model (the physics-affecting kwargs and how to choose a caustic mode); routing; **the
measurement recipe**; known limits; the claim -> test-pin table (14 rows, every path
checked to exist); and a maintenance rule.

VERIFY-A2's two required items are in:

* **the sampling recipe** (§6): `dx ~ 1.45 * aperture / 2048`, not a carrier-Nyquist
  pitch; at N = 512 a meniscus reads `model - wave = 557.8 nm` and an air-spaced doublet
  949.2 nm of pure hard-edge aliasing through the in-glass ASM, converging to 0.03 nm by
  N = 2048 (`|E|` ripple 0.640-1.130 -> 1.052-1.057).  The five-fixture convergence table
  is included, and the through-focus caveat (two refinement passes, step <= 0.1 DOF).
* **the L1 fast-element caveat and its clamp fix** (§3.2 and §7 item 1).  The clamp has
  LANDED (VERIFY-A2 follow-up F.1: fan launched to `+-0.999 r_pupil`, screen held at
  `corr(rho_fit)` outside `rho_fit`), and the doc carries the post-clamp numbers
  (8 mm doublet 173.6 -> 1.053 nm / 165x; f/2 402.6 -> 1.504 nm / 268x) *and* F.1's own
  retraction: the residual -36.6 % peak on the f/2 is not extrapolation, it is that the
  fixture's exit field does not fill the pupil the radial screen is normalised to
  (central-row energy 94.59 % inside `rho <= 0.85`).  The doc says no guard ships and
  why: `rho_fit` does not separate that case from the 8 mm doublet (`rho_fit = 0.783`,
  correction excellent).

Both fenced blocks execute; the first prints the 0.118 mm of sag a naive `image_rays`
read would carry.

---

## 6. D5 — README

* **Start here** table (audit report + per-partition reports + repro scripts + the
  `RESOLUTION_STATUS.md` table being generated at release; the subsystem doc; the
  Migration-Guide's v5.46 section; CONVENTIONS; CHANGELOG; ROADMAP), plus two lines
  stating the power convention and metres-everywhere.
* **CODE V units at the old 3177-3179** rewritten exactly as WP-A10 §5.4 asked, with the
  `DIM M` = millimetres statement, the pre-v5.46 detection + warning, and `dim_units=`.
* **`LGAberrationMerit`** block annotated with the v5.46 scale change and a link to the
  migration note; the constructor was executed to confirm the shown arguments still work
  (and that `strehl_branch` defaults to `'sigma'`).
* Identifier rewrites listed in §2.3.

**Snippet execution** (`<scratch>/wpa18/run_snippets.py`, which extracts every fenced
`python` block — including indented ones — and executes it):

| file | blocks | pass | skip | fail |
|---|---:|---:|---:|---:|
| `README.md` | 14 | 10 | 1 | 3 |
| `docs/subsystems/real_lens.md` | 2 | **2** | 0 | 0 |
| `Migration-Guide.md` (the new `override` block) | — | **1** | — | — |
| `docs/cookbook.md` (with a shared preamble) | 17 | 10 | 0 | 7 |

The three README failures are **pre-existing illustrative fragments**, all in
"What's new in 3.4.0" / "3.3.x" / "4.11" historical sections, none of whose *code* I
edited: one is a deliberate sketch
(`design_optimize(parameterization=..., jac='auto', ...)` — a bare `...` after a keyword
argument, so a `SyntaxError`), one is a two-line fragment assuming a prior `E_in`, and
the third is the indented `LGAberrationMerit` block, which assumes a prior
`import lumenairy as la`.  That third one is the block I annotated, so I executed its
body with the import supplied: it constructs, and `strehl_branch` reads `'sigma'`.  The
seven cookbook entries — the blocks the README presents as runnable — all pass.
`docs/cookbook.md`'s seven failures are likewise fragments that need a `.zmx` file or a
prior `prescription` / `E1` binding; **none is a changed-default breakage**, which is what
I was looking for.

The Migration-Guide's other 15 "failures" are the OLD halves of before/after migration
pairs — `from lumenairy.elements.rcwa import apply_thin_grating`,
`import lumenairy.system`, and twelve fragments assuming a prior `import lumenairy as la`.
They must not run; that is the point of a migration guide.  The one block I added (the
`override` context manager) passes.

---

## 7. Tests run

```
OPENBLAS_NUM_THREADS=1 python -m pytest <files> -q --no-header -p no:cacheprovider
```

| batch | files | result | duration |
|---|---|---|---|
| 1 | `test_audit_v5_24_2_g01_conventions.py`, `test_v4_16_1_agent_d.py`, `test_v4_16_2_dispatcher_pin_doc_consistency.py`, `test_v4_16_2_agent_d.py`, `test_niche_p8_capstone.py` | **63 passed** | 118.2 s |
| 2 | `test_v4_15_2_agent_e.py`, `test_v4_15_3_agent_d.py`, `test_v4_15_4_agent_c.py`, `test_v4_15_5_agent_c.py`, `test_v4_15_agent_f.py`, `test_v4_16_3_agent_a.py`, `test_v4_16_3_agent_b.py`, `test_v4_16_3_agent_d.py` | 100 passed, **1 failed** | 4.4 s |
| 3 | `test_g08_s4_20_packaging_ui.py`, `test_g10_s5_9_layerspec_tracked.py`, `test_v5_2_walker_changelog_changeset.py`, `test_v5_4_1_walker_scope_the_workaround.py`, `test_validation_helpers.py`, `test_v5_2_5_ao_closed_loop_residuals.py` | **75 passed, 2 skipped** | 8.6 s |
| 4 | `test_audit2609_a12_pmm1d.py::test_g4_the_1d_jones_is_the_lab_cartesian_basis_and_jxx_is_minus_rp` | **1 passed** | 0.8 s |

The file set is `grep -rln "README\.md\|Migration-Guide\.md\|CONVENTIONS\.md\|ROADMAP\.md"
tests/unit` (25 files), minus the heavy solver files whose doc reference is a prose
citation rather than a file read.  The two skips are `test_v5_2_walker_changelog_changeset.py`
declining to reconcile a `## [5.45.1]` CHANGELOG block that declares no claims.

### The one failure, and why it is not mine

`test_v4_16_3_agent_d.py::test_propagation_documents_multiprocess_fork_semantics` reads
`lumenairy/propagators/fft_infra.py` and `propagation.py` — **no markdown** — and asserts
that one of them contains the string `Multiprocess / fork notes`.  Measured:

```
git show HEAD:lumenairy/propagators/fft_infra.py | grep -c "Multiprocess / fork notes"   -> 1
grep -c "Multiprocess / fork notes" lumenairy/propagators/fft_infra.py                   -> 0
git status --porcelain lumenairy/propagators/fft_infra.py                                -> " M"
```

i.e. the section is present at `HEAD` and has been removed by an **uncommitted** edit to
that file by another work package (this is the shape of the audit V6 / WP-A17
history-block relocation).  Owner: whoever holds `lumenairy/propagators/fft_infra.py`.
Either restore the `Multiprocess / fork notes` heading over the surviving `DEFAULT_*`
discussion, or retire this test with the rest of the source-prose tests WP-A15a started
removing (audit V5: 211 tests assert on `__doc__` / `inspect.getsource`).

---

## 8. Requested changes outside my ownership

1. **`lumenairy/propagators/fft_infra.py` (WP-A15b / WP-A17)** — restore the
   `Multiprocess / fork notes` section header, or coordinate with the owner of
   `tests/unit/test_v4_16_3_agent_d.py` to retire
   `test_propagation_documents_multiprocess_fork_semantics`.  See §7.
2. **`lumenairy/elements/rcwa/twod.py:1301, :1303` (WP-A14)** — the Wood-anomaly
   diagnostic passes `fn_name="RCWA2DPrepared.solve"`, but the class is
   **`PreparedRCWA2D`** (`twod.py:1259`, built by `prepare_rcwa_2d`).  A user who greps
   the warning text for the class finds nothing.  One-word change in two string literals;
   my Migration-Guide note documents both spellings so the guide is correct either way.
3. **`lumenairy/ui/lens_options_dialog.py` (UI WP)** — WP-A3 §5.8 already raised this and
   it is still open: the `fast_analytic_phase` tooltip says "~25 % speedup with < 10 nm
   OPL error"; the measured error is ~7 nm rms **per mm of glass**.  I have recorded the
   per-mm form in `docs/subsystems/real_lens.md` §4, so the doc and the tooltip now
   disagree until the tooltip is fixed.
4. **`CHANGELOG.md` / `GUI_CHANGELOG.md`** — assembled by the release step from
   `fixes/*_CHANGELOG.md`; mine is `fixes/WP-A18_CHANGELOG.md`.
5. **`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/RESOLUTION_STATUS.md`** —
   `README.md`, `Migration-Guide.md` and `ROADMAP.md` now link to this path.  It does not
   exist yet (it is generated at release).  If the release generates it somewhere else,
   three links need updating; `docs/audits/` is outside my ownership so I did not create
   a placeholder.

## 9. Deferred

* **Splitting the README** (audit P3-2: 180 KB, ~3 700 lines, mostly embedded per-version
  release notes that the file itself says went stale after v5.1.0).  The audit's plan is a
  ~10 KB landing page plus `docs/cookbook/` run as doctests.  My brief says "split nothing
  large in this pass", and the landing table is the navigational half of that work.
  Effort: ~1 d for the split, ~1 d to convert the cookbook to doctests and retire the
  remaining prose tests.  Doing it in the same pass as eleven other work packages' edits
  would have made every reviewer's diff unreadable.
* **`docs/audits/` archiving by year** and the CHANGELOG split — outside my ownership
  (`docs/audits/`) and a release-step decision respectively.
* **A committed resolver.**  The identifier resolver is the kind of thing that should run
  in CI (it is ~250 lines and needs no network), but it is a `.py` file and I own none.
  If the project wants it, it belongs in `scripts/` beside `check_source_line_citations.py`
  with a test that asserts the unresolved count is 0.  Its rules are documented in §2.1 so
  it is reproducible from this report.

## 10. Late-arriving inputs (the orchestrator's re-list)

The `fixes/` directory was snapshotted at the start of this pass and re-listed at the end:
**no new `*_CHANGELOG.md` file appeared**.  Two were **modified** after I read them —
`WP-A4_CHANGELOG.md` and `VERIFY_WP-A4_CHANGELOG.md`, both re-stamped 10:12 — and I
re-read both; the additions (the LG merit's dimensionless form, the
`van_vleck_weight` round-trip correction, `aberration_free_reference_fit`, the Maslov
vector wrapper's joint normalisation) are all carried in the new migration section.

**Still absent, and their notes are therefore NOT in `Migration-Guide.md`:**

| expected file | what the guide will need |
|---|---|
| `WP-A16_CHANGELOG.md` (lens config objects) | if `LensGeometry` / `LensNumerics` / `LensResources` land with a deprecated kwarg shim, that is a first-order migration note, and `ROADMAP.md`'s leading table should move that row from "remaining" to "done".  The five `_optional` call-site switches and the lens-family knob registrations (WP-A15b §5.4, §5.5) would add two sentences to CONVENTIONS §10.1. |
| `WP-A17_CHANGELOG.md` (history relocation) | not user-facing, but if it removes a docstring section that a test asserts on — which is exactly the failure in §7 — the guide should say that source-prose tests were retired with it. |
| `WP-A21_CHANGELOG.md` (hygiene) | anything that moves a shipped path (`MANIFEST.in`, `scripts/` -> `validation/probe_scripts_legacy/`) needs a line in the packaging group. |

Ownership note: `GUI_README.md` is on my list "if the UI WP requested it".  WP-A9 made its
two `GUI_README.md` edits itself (commit `58ad836c`: the source table's object-distance
note and the form-field tooltip) and its report asks nothing further of me, so I left the
file alone.

---

## 11. Path to the changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A18_CHANGELOG.md`

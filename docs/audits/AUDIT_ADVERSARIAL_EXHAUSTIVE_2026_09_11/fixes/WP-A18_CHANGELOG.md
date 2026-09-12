# WP-A18 changelog text — documentation (README / Migration-Guide / CONVENTIONS / ROADMAP / subsystem doc)

Audit findings **V7** (repo hygiene / doc identifiers, TESTS-ARCH **P3-2**) and the
"docs consolidation strategy" of TESTS-ARCH, plus the documentation requests raised by
WP-A2 (§5.1, §5.2), WP-A5 (§5.3), WP-A10 (§5.3, §5.4), WP-A13 (§5.2), WP-A14 (§5c) and
WP-A15a (§5.11).

---

### Added -- docs: `docs/subsystems/real_lens.md`, the living contract of the `apply_real_lens` family

The first of the per-subsystem documents the audit recommends (section 14 V7: "archive
`docs/audits/` by year and keep one living document per subsystem").  343 lines covering:
the nine entry points and when to use each; the one invariant that binds them (the
exit-vertex plane, audit section 15.1) with a runnable demonstration that a plano-convex
singlet leaves its rays **0.118 mm** off the vertex plane; what every physics-affecting
option of `apply_real_lens` and `apply_real_lens_traced` does NOW, with the measured
envelope and the finding ID; the `tilt` key; how to choose a caustic mode; the router's
frame-invariance fix; the **measurement recipe** (`dx ~ 1.45 * aperture / 2048` for an
exit-OPL measurement -- at N = 512 a meniscus reads 557.8 nm of pure hard-edge aliasing
through the in-glass ASM, converging to 0.03 nm by N = 2048); seven known limits; and a
table mapping every claim to the test that pins it.

The known-limits section records the two scope limits the verification passes found and
did not close: `seidel_correction` on an **under-filled pupil** (f/2 biconvex: exit
wavefront 268x better inside `rho <= 0.85`, focal peak ~37 % lower, best focus 8.2 DOF
away -- and `rho_fit` does NOT separate that case from the 8 mm cemented doublet where
the same flag is excellent, so no guard ships), and **FGA convergence through a real
singlet** (intensity-rms width 12.721 um against `phase_screen`'s 3.269 um and the
real-ray fan's 3.44 um, at tilt 0 as well as under tilt).

Both code blocks in the file execute.

### Changed -- docs: `CONVENTIONS.md` section 7 states the power convention, the implemented tilt convention and the Richards--Wolf pupil coordinate

* **Power (audit L13).**  New table row + bullet: `sum(abs(E)**2) * dx * dy` **IS** the
  optical power -- the propagators are Parseval-unitary and carry no impedance factor --
  so any element crossing an index step must apply the POWER transmittance
  `T = (n2 cos theta_t)/(n1 cos theta_i) abs(t)**2`, not `abs(t)**2`.  Measured
  AIR -> N-BK7: 0.632344 against 0.958057 (and the closed form
  `4 n1 n2/(n1+n2)**2` to 1.1e-16).  The convention `fresnel=True`'s fix depends on was
  written down only inside `_lens_real.py`.
* **Tilt (audit L2 / L19).**  The old row claimed
  `tilt=(theta_x, theta_y, theta_z)`, a spelling **no consumer of that key has ever
  implemented**.  It is replaced by the implemented convention: `tilt = (t0, t1)` IS the
  linear sag ramp `t0*(x - dcx) + t1*(y - dcy)` added to the surface's z-departure, with
  the correspondingly tilted normal; as a right-hand rotation pair, `theta_x = t1` and
  `theta_y = -t0`.  The row names its five consumers (the `surface_frame=False` ramp,
  the `surface_frame=True` rigid-body branch, `_disp_surface_z_grad`,
  `raytrace/surface.py::_field_frame_sag_and_grad` and the lumenairy-free
  `geom_spot_decenter_oracle`) and the cross-model pin
  `tests/unit/test_niche_p9_decenter_tilt.py`.  A bullet after the table says explicitly
  that a `(theta_x, theta_y, theta_z)` spelling describes the *coordinate-break* keys
  `tilt_x_deg` / `tilt_y_deg` / `tilt_z_deg` one row down, which are a different key with
  a rigid-frame meaning.  **No stored `tilt` value needs migrating.**
* **Richards--Wolf `pupil` (audit K10 / K16).**  New row: indexed by the physical
  exit-pupil (aperture) coordinate, NOT the projected ray direction; the two differ by a
  point inversion -- invisible for a 180-degree-symmetric pupil, immediately wrong for
  coma, tilt, a decentred sub-aperture, a segmented aperture or a metasurface pupil.
* **Section 7.1 (audit G4/G13)** -- WP-A12's replacement text was reviewed and kept
  unchanged; it already states the index ORDER explicitly (index 0 = x = p/tm at
  `phi = 0`, index 1 = y = s/te) and `J[0,0] = -r_p`, which is what WP-A13 section 5.2
  asked for.  Recorded in the file's own changelog so the decision is not re-litigated.
* **Section 11 (audit H3 / G6)** -- records `rcwa_jones_2d(formulation='fff_nv')`'s new
  validated-scope warning on a curved cell and its `allow_nonseparable_nv` escape, and
  `pmm_jones_2d`'s new `formulation='auto'` with the observable-dependent accuracy
  ordering (the docstring ordering is the REFLECTION one; on the TRANSMISSION retardance
  `'li'` is the most accurate of the three at `n_orders >= 9`).
* **Section 10 / new 10.1 (audit H5, V6)** -- `threadpoolctl` is a HARD dependency, not
  an optional one (measured: a 1-D TM RCWA solve at `n_orders=81` takes 18.2 s unpinned
  against 0.13 s at one thread), and process globals are scoped with
  `lumenairy.override(**knobs)`, with the rule for registering a new knob.

Every entry above is recorded in the file's "Changelog of this file" section.

### Added -- docs: `Migration-Guide.md` gains the v5.46 audit-remediation section

~650 lines, eleven area groups (analytic lens; traced lens and caustic siblings;
Maslov / GBD / FGA and the asymptotic family; propagator kernels; traced-carrier chain;
analysis metrics; thin elements and materials; polarization / sources / algebra / memory;
rigorous grating solvers; I/O, prescriptions and the optimiser; designer UI and the lens
library; packaging, configuration and public API).  Every note names its finding ID, the
old and the new behaviour with the measured numbers, and the way back where one exists
(`method='auto'` on `FreeSpace`, `min_feature=period*1e-5`, `symmetry=False`,
`normalisation='legacy'`, `vector_projection=False`, `kernel='spatial'`, `thin_form=True`,
`pad_sigma=0.0`, `dim_units=`, `compression='gzip'`, `r0 * 2**(-3/5)`, an explicit
`standoff=`, an explicit `rng=`, `dtype=np.float32`, `on_undersample='silent'`,
`integration_method='local_quadrature'`) -- and says so plainly where there is none
because the old answer was simply wrong.

The notes are taken in substance from the WP changelog files; nothing is invented.  The
v5.2.0 `surface_frame` section gained a superseded-in-part banner pointing at the new
section, and the stale "This guide covers v4.x" preamble was corrected.

### Changed -- docs: README landing section, CODE V units, and the identifier sweep

* A **Start here** table at the top routes the reader to the cookbook, the
  Migration-Guide's v5.46 section, `CONVENTIONS.md`, the new subsystem document, the
  audit report + per-partition reports + repro scripts + the resolution table, the
  CHANGELOG and the ROADMAP -- with the power convention and the metres-everywhere rule
  stated in two lines before the first call.
* **CODE V units (audit I1, WP-A10 section 5.4)**: "units M/MM/IN" is corrected to
  "units **M (mm) / C (cm) / I (inch)**, with `MM`/`CM`/`IN` tolerated as aliases on
  read", and the entry now says that `DIM M` is CODE V's millimetre token, that files
  lumenairy wrote before v5.46 carry SI metres under it and are detected and warned
  about, and that `dim_units=` forces a reading.
* The `LGAberrationMerit` cookbook block carries a v5.46 scale-change note pointing at
  the migration note (the merit is now a dimensionless coupling and gained
  `strehl_branch`).  The block was executed to confirm the constructor still accepts the
  arguments shown.
* **Non-resolving identifiers (audit V7 / P3-2).**  A resolver over every backticked
  identifier in `README.md`, `ROADMAP.md`, `Migration-Guide.md` and `CONVENTIONS.md`
  (1 952 occurrences / 1 055 distinct identifier-shaped tokens) against the CURRENT
  package -- dotted paths, module attributes, class members, nested definitions, and a
  static AST index for `lumenairy/ui/` (which needs PySide6) -- now reports **0
  unresolved** of a 592-token API-claiming denominator, against 4 before this pass.  Of
  the twelve identifiers the audit named, five already resolved against a deep module
  (`_decompose_prescription`, `_detect_backend`, `_PROPAGATE_SYSTEM_JAX_CACHE`,
  `_spawn_rng`, `_fd_grad_pure` -- the audit resolved only against `lumenairy` and its
  nine subpackages) and are now cited by their module path where the bare name was
  ambiguous; three are real kwargs (`return_kind`, `opl_fn`, `image_centres`); two are
  `raytrace.Surface` fields (`world_origin`, `world_R`); `focus_fixed_sampling` is
  **prysm's** API named as a comparison (now stated in the text, along with POPPY's
  `apply_image_plane_fftmft`); `rcwa_1d` is documented as removed and the second
  citation now says so and names `rcwa_efficiency_1d`; `fiber_mode` is a source-kind
  string and is now quoted as one.  Four genuinely stale references were rewritten:
  `row_reset` / `last_v_star` (a dead local name; now written as the
  `maslov_tracking='row_reset'` option it describes),
  `raytrace._intersect_surface` -> `raytrace.intersection._intersect_surface`,
  `lumenairy.asymptotic` -> `lumenairy.propagators.asymptotic` in the feature list
  (with the v5.0 relocation stated at the historical mention), and a test-class citation
  replaced by prose.

### Changed -- docs: `ROADMAP.md` leads with the audit backlog

The file opened with "There is no open ROADMAP code-work as of v5.4.7" and a
2026-05-30 date.  A new leading section dates it 2026-09-12, points at the audit report
and the resolution table, and tabulates what remains after the v5.46 physics and data
remediation with the audit's own effort estimates: the lens-family config objects (10 d),
history blocks out of the source (5 d, -12 484 lines across six files), a leaf
`elements/_lens_kernels.py` (3 d), docs consolidation (3 d -- started here), the pairwise
covering array (3 d), `probe_*` out of `validation/` (2 d), the section 15.4
consolidation targets and the section 15.9 alternative algorithms.  It also records the
process rule the audit calls highest-leverage: an audit round may not add a new test
file; it must strengthen the existing test for that kwarg or explain why one does not
exist.  The historical v5.4.x content is kept below, unchanged.

---

**Files:** `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`, `ROADMAP.md`, new
`docs/subsystems/real_lens.md`.  No source file and no test file was modified.

**Verification:** the repository's doc-reading tests -- **216 passed, 2 skipped, 1 failed**
across `test_niche_p8_capstone.py`, `test_audit_v5_24_2_g01_conventions.py`,
`test_v4_16_1_agent_d.py`, `test_v4_16_2_agent_d.py`,
`test_v4_16_2_dispatcher_pin_doc_consistency.py`, the six `test_v4_15*` / `test_v4_16_3*`
agent files, `test_g08_s4_20_packaging_ui.py`, `test_g10_s5_9_layerspec_tracked.py`,
`test_v5_2_walker_changelog_changeset.py`, `test_v5_4_1_walker_scope_the_workaround.py`,
`test_validation_helpers.py`, `test_v5_2_5_ao_closed_loop_residuals.py` and
`test_audit2609_a12_pmm1d.py::test_g4_the_1d_jones_is_the_lab_cartesian_basis_and_jxx_is_minus_rp`.
The one failure, `test_v4_16_3_agent_d.py::test_propagation_documents_multiprocess_fork_semantics`,
reads only `.py` files and is **not attributable to this work package**: the
`Multiprocess / fork notes` section it asserts on is present in
`lumenairy/propagators/fft_infra.py` at `HEAD` and absent from the uncommitted working
tree of that file, which another work package is editing.  Every fenced `python` block
in the seven cookbook entries, the new subsystem document and the new migration section
executes.

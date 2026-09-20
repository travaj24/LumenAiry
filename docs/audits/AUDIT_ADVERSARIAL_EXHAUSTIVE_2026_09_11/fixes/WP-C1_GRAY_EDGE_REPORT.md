# WP-C1 -- `apply_aperture(edge='gray')` becomes the default

Built 2026-09-20 on `49ddf4bd` (branch `feat/c1-gray-edge-default`).  The decision
is the maintainer's, recorded in
[`../MAINTAINER_DECISIONS_2026_09.md`](../MAINTAINER_DECISIONS_2026_09.md)
section 1.1 on the measurement in
[`WP-B11_REPORT.md`](WP-B11_REPORT.md) section 2.9.  This work package
implements it: it re-measures the case on an independent build of the probe,
flips the default, finds every answer that moves, re-pins the tests that move
against their own oracles, and proves the way back is exact.

Everything below was measured on BOTH builds -- Windows py3.14 / numpy 2.4.4 /
scipy-openblas and WSL py3.12 / numpy 2.4.6 / scipy-openblas -- with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `lumenairy.__file__` printed by every probe, and byte comparisons taken
ARCHIVE TO ARCHIVE against a `git archive 49ddf4bd` extraction, never against a
working copy.  Probes and their JSON: `validation/probe_c1_gray_edge/`.

---

## 0. A premise the task's own statement got wrong, checked first

The brief named `49ddf4bd` as "main, the fully merged 5.48.1 tip".  It is not
the 5.48.1 tip: the 5.48.1 release commit is `e995f00e`, and `49ddf4bd` is the
merge ONE commit later (`origin/main`; the local `main` ref in this checkout was
stale at `e995f00e`).  The two differ only in two handoff documents
(`HANDOFF_2026_09_14.md` and `handoff/orchestrator_log_2026_09_13-14.md`, 4
insertions and 6 deletions), so no library file and no fixture is affected.  The
worktree was reset to `49ddf4bd` exactly as the brief asked, and every byte
comparison in this report is against `git archive 49ddf4bd`.

---

## 1. The measurement, re-run before anything was changed

`validation/probe_c1_gray_edge/probe_ladder.py`.  Geometry and oracle are
WP-B11 section 2.9's: lambda = 633 nm, circular aperture of radius a = 100 um
centred on a 512 um window, N = 128 / 256 / 512 / 1024, and the error is the
on-axis relative error against the CLOSED FORM

    U(0, 0, z) = e^{ikz} - (z / r_a) e^{ik r_a},    r_a = sqrt(z^2 + a^2)

-- the Rayleigh-Sommerfeld integral evaluated on axis, which depends on nothing
discrete.  Two kernels: `rayleigh_sommerfeld_propagate(kernel='spatial')` at
z = 16 mm (above that kernel's own alias threshold `2 N dx^2 / lambda` at every
N here) and `propagate_huygens_fresnel_with_opl_callable` at z = 5 mm with the
exact spherical OPL, evaluated at ONE on-axis output point.

The probe runs THREE arms: `edge='hard'`, `edge='gray'`, and the DEFAULT arm
called with no keyword at all, and it records whether the default arm is
BIT-IDENTICAL to one of the named two.  That third arm is the entire content of
this work package, measured rather than asserted.

### 1.1 Before the flip (`ladder_BEFORE_WIN.json`, `ladder_BEFORE_WSL.json`, run against the `git archive` tree)

| N | RS hard | RS gray | RS default | HF hard | HF gray | HF default |
|---|---|---|---|---|---|---|
| 128 | 8.3008e-03 | 1.4847e-03 | 8.3008e-03 | 2.7708e-02 | 1.4438e-02 | 2.7708e-02 |
| 256 | 3.3548e-03 | 3.6098e-04 | 3.3548e-03 | 1.1120e-02 | 3.4928e-03 | 1.1120e-02 |
| 512 | 3.4207e-04 | 8.1013e-05 | 3.4207e-04 | 1.1509e-03 | 8.5680e-04 | 1.1509e-03 |
| 1024 | 5.2718e-04 | 2.5030e-05 | 5.2718e-04 | 1.7601e-03 | 2.1122e-04 | 1.7601e-03 |
| order | 1.307 / 3.294 / **-0.624** | 2.040 / 2.156 / 1.694 | (= hard) | 1.317 / 3.272 / **-0.613** | 2.047 / 2.027 / 2.020 | (= hard) |

`default is bit-identical to: ['hard']` on both kernels and both builds.

**Every one of WP-B11 section 2.9's sixteen table entries is reproduced to the
last printed digit**, on an independently written probe, on two builds.  That is
the verification of the measurement the decision rests on.

### 1.2 After the flip (`ladder_AFTER_WIN.json`, `ladder_AFTER_WSL.json`)

| N | RS hard | RS gray | RS default | HF hard | HF gray | HF default |
|---|---|---|---|---|---|---|
| 128 | 8.3008e-03 | 1.4847e-03 | 1.4847e-03 | 2.7708e-02 | 1.4438e-02 | 1.4438e-02 |
| 256 | 3.3548e-03 | 3.6098e-04 | 3.6098e-04 | 1.1120e-02 | 3.4928e-03 | 3.4928e-03 |
| 512 | 3.4207e-04 | 8.1013e-05 | 8.1013e-05 | 1.1509e-03 | 8.5680e-04 | 8.5680e-04 |
| 1024 | 5.2718e-04 | 2.5030e-05 | 2.5030e-05 | 1.7601e-03 | 2.1122e-04 | 2.1122e-04 |
| order | 1.307 / 3.294 / **-0.624** | 2.040 / 2.156 / 1.694 | (= gray) | 1.317 / 3.272 / **-0.613** | 2.047 / 2.027 / 2.020 | (= gray) |

`default is bit-identical to: ['gray']`, and **the `hard` column is unchanged in
every digit** -- the way back is not merely close.

### 1.3 The decision the default rests on, stated build-free

Not "the error is 2.5030e-05", which is a reading.  The claims the shipped test
file asserts, each with its measured headroom:

| claim | bar | RS measures | HF measures |
|---|---|---|---|
| the grey arm falls at EVERY refinement | monotone | yes | yes |
| ... and by at least 32x over the ladder | >= 32 | 59.3x | 68.4x |
| the hard arm does NOT fall at every step | at least one step up | 3.4207e-04 -> 5.2718e-04 (+54 %) | 1.1509e-03 -> 1.7601e-03 (+53 %) |
| the hard arm has a NEGATIVE order, the grey arm none | sign | -0.624 | -0.613 |
| the grey arm wins at the finest grid | >= 5x | 21.1x | 8.3x |

Both builds read every entry in 1.1 and 1.2 to five significant figures, so the
cross-build spread of these quantities is below the last printed digit while the
decisions turn on factors of 1.5 and more.  The +54 % rise is four decades
outside that spread.

### 1.4 `edge_samples` stays at 4, and 4 is now pinned as the knee

RS ladder, N = 512, the same geometry (2026-09-20, Windows; WSL agrees to five
figures):

| `edge_samples` | 1 | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|
| on-axis rel. error | 3.4207e-04 | 1.6563e-04 | **8.1013e-05** | 8.3954e-05 | 8.3959e-05 |

Two-sided: 2 -> 4 still gains **2.04x** (bar 1.5x, 1.36x of headroom) and 4 -> 8
gains **0.965x**, i.e. it is 3.5 % WORSE (bar "less than 1.1x", 1.14x of
headroom), because past the knee the residual is the propagator's, not the
mask's.  `edge_samples=1` reproduces the hard reading to the last bit -- n_sub = 1
IS the pixel-centre indicator -- and the shipped test asserts that identity, so
the ladder cannot drift into measuring something else.

Cost, quoted from WP-B11 and unchanged: the extra indicator evaluations touch
only boundary pixels, 312 of them at N = 256 (0.476 % of the grid) and 1196 at
N = 1024 (0.114 %), i.e. 0.076x and 0.018x of one full-grid pass.

---

## 2. What was changed

### 2.1 The default

`lumenairy/elements/elements.py`: `apply_aperture(..., edge='gray')`.  The
docstring now carries the convergence table, the statement that the hard arm has
no order at all, the cost, and the sentence that `edge='hard'` reproduces the
pre-5.49 answer bit for bit.  The module header's "hard-edge amplitude masks"
was restated as "sharp-edged (unapodized)", because after this change
"hard-edge" is a keyword value and no longer a synonym for "unapodized".

### 2.2 In-library callers: how many needed `edge='hard'` made explicit?

**None**, which is what the brief expected.  The five in-library call sites and
why each correctly takes the new default:

| caller | what it is | verdict |
|---|---|---|
| `elements.apply_lyot_stop` | coronagraph-named annular stop | "hard-edge" in its docstring meant UNAPODIZED (contrast `apply_apodized_pupil`), not "staircase".  A grey rim is the same physical stop rendered better.  Docstring restated; it now says its answer moved and names the way back. |
| `algebra/apertures.py::Aperture._apply` | the algebraic surface | deliberately minimal vocabulary; no binary-mask contract anywhere in it.  Notes restated. |
| `elements/polarization.py::JonesField.apply_aperture` | per-component stop | no contract beyond "apply the stop to both components".  Docstring added naming the move and the way back. |
| `propagators/system.py`, `'aperture'` element | chain element | its schema said "Hard aperture"; that was the same unapodized sense.  Restated, and the element gained `edge` / `edge_samples` keys (below). |
| `io/codegen.py` STOP surface | emits `la.apply_aperture(E, dx, shape="circular", params=...)` as TEXT | deliberately emits no keyword, so a generated script tracks the library default exactly as the chain it was generated from does.  Pinning either value into generated source would freeze today's default into every future script.  Left alone; listed in the Migration note. |

### 2.3 The way back had to EXIST at every entry point, and for a chain it did not

`propagate_through_system` forwarded no `edge`, so after the flip a chain caller
had no way to ask for the old rim at all.  Both chains now read optional
`'edge'` / `'edge_samples'` element keys through one helper,
`system._aperture_edge_kwargs`, which returns only the keys the element actually
names -- so an element that names neither cannot pin today's defaults into
tomorrow's answer.

### 2.4 The JAX twin's own copy of the mask, and a defect it was hiding

`propagate_through_system_jax` carried its own pixel-centre indicator, in TWO
places (the jit'd kernel and the `verbose=True` slow path).  Left alone, the
flip would have made the NumPy chain and the JAX chain answer the same element
dict differently, and the existing cross-backend test
(`test_audit_misc.py::test_jax_aperture_matches_numpy`) would NOT have caught it:
its bar is "fewer than 5 % of pixels mismatched", and a rim on a 64-pixel disk is
about 1.5 %.  A silent backend divergence introduced by this change.

Both routes now call `elements.apply_aperture` -- which is
`backend.array_namespace`-dispatched and traces under `jax.jit` and `jax.grad`
(measured: jit OK, grad OK, complex64 preserved) -- so there is one
implementation, one default and one way back.  `_system_element_signature` now
carries `edge` and `edge_samples` in the STATIC kernel signature, so two elements
that differ only in the rim are two compiled kernels rather than one silently
serving both.

**The defect this uncovered predates the release.**  The slow path multiplied by
a boolean mask (`E * mask.astype(E.dtype)`), so a blocked pixel came back as a
SIGNED zero.  Measured at `49ddf4bd` on a 128 x 128 fixture: **2652 negative-zero
real parts and 2762 negative-zero imaginary parts**, where the jit'd kernel (XLA
rewrites the product into a select) and the NumPy chain both returned `+0.0`.
`np.array_equal` reports those arrays as equal, and so does every tolerance test;
only the byte comparison sees it.  The same multiply left a non-finite field
non-finite outside the stop -- the defect VERIFY-A8 fixed on the NumPy path and
which had never reached the JAX one.  Both routes now select rather than scale
and are byte-identical to the parent commit's NumPy answer.

### 2.5 History fingerprints

Two modules changed token stream and were re-recorded IN THIS CHANGE with their
reasons: `lumenairy.elements.elements` and `lumenairy.propagators.system`.
`lumenairy.elements.polarization` and `lumenairy/algebra/apertures.py` were
touched in docstrings only, and the gate proves it -- they do not drift.
`python scripts/record_history_fingerprints.py --check` reports every history
document matching its module.

---

## 3. Blast radius, measured

Selection: every file under `tests/unit` matching `apply_aperture`,
`apply_lyot_stop`, `Aperture(`, or a quoted `'aperture'` (the chain element) --
**33 files, 1394 tests**.  Run before the change (`1394 passed` in 7:33) and
after (`4 failed, 1390 passed` in 9:42), both with `-p no:randomly
--capture=sys -q`, so every failure below is a MOVE, not a pre-existing red.

### 3.1 The four tests that failed

| test | classification | what was done |
|---|---|---|
| `test_audit2609_a8_thin_elements.py::test_e7_aperture_hard_edge_is_the_default_and_unchanged` | **(a)** a pin of the old default | Renamed to `..._is_one_keyword_away_and_unchanged`.  The "unchanged" half keeps its exact binary-mask comparison and now names `edge='hard'`; the default assertion is RE-DECIDED to `'gray'`, the value the maintainer ruled for.  It additionally asserts, derived from the fixture rather than stated, that the default differs from the binary mask exactly where the rim CUTS a pixel -- true for the two curved rims and false for this rectangle, whose rims land on pixel boundaries (37.5 dx and 22.5 dx), so grey and hard coincide there exactly. |
| `test_audit2609_a8_thin_elements.py::test_e7_aperture_gray_edge_removes_the_area_quantisation` | **(c)** a genuine contract on the hard edge | It is the hard-vs-gray comparison; its `h` reading was the unnamed default.  Now `edge='hard'` explicitly.  **Readings and bars unchanged.** |
| `test_audit2609_a8_verify.py::test_verify_a8_e7_gray_edge_is_not_advertised_for_axis_aligned_rims` | **(c)** | Same shape: `assert e_g4 < abs(_area() / analytic - 1)` had become `e_g4 < e_g4`.  Now `_area(edge='hard')`.  Bars unchanged. |
| `test_v4_15_1_agent_g_application.py::TestApertureApplication::test_circular_aperture_zeros_outside_disk` | **(b)** a fixture now closer to the closed form, whose claim had to be restated one level up | Renamed to `test_circular_aperture_renders_the_disk_by_pixel_area` and restated from the pixel CORNERS, which says strictly more than the old pair of bars did: a pixel whose farthest corner is inside passes EXACTLY (where the old test allowed `allclose`), a pixel whose nearest corner is outside is EXACTLY zero (where the old test allowed `< 1e-30`), every rim pixel is the input scaled by a real factor in [0, 1], at least one pixel is in each of the three sets, and -- the reason the default moved -- the grey transmitted area is closer to the analytic disc area than the hard one, measured here both ways as a decision with no bar.  **No bar was loosened; two were tightened to exact.** |

### 3.2 The two tests that did NOT fail and had to be fixed anyway

Adversarial in the other direction: a moved default can also turn an assertion
into a tautology, which is green and meaningless.

| test | what happened | what was done |
|---|---|---|
| `test_audit2609_a8_verify.py::test_verify_a8_e7_gray_edge_beats_hard_at_anamorphic_and_offset_rims` (parametrised, 3 cases) | `e_hard = abs(_area() / analytic - 1)` silently became the GREY reading, so `assert e_g4 <= e_hard` became `e_g4 <= e_g4` -- **green, and no longer a comparison**. | `_area(edge='hard')`.  Bars and readings unchanged. |
| `test_verify_a8_e7_gray_edge_zeroes_a_blocked_pixel_even_if_it_is_not_finite` | its variable named `hard` had become the grey arm, so the test asserted the grey contract twice | `edge='hard'` named explicitly. |

### 3.3 Tests that touch an aperture and correctly did not move

The remaining 27 selected files are green unchanged.  Spot-audited for vacuity
rather than trusted: the `Aperture` ABCD / repr tests (`test_v4_15_1_agent_g_abcd.py`,
`test_v5_4_6_wave5_delegated.py`) assert identity matrices and reprs, which no rim
touches; `test_v4_15_1_agent_g_examples.py` asserts a focusing ratio > 50 on a
chain containing an aperture, whose subject is the focus; `test_audit_io.py`
asserts the codegen EMITS an `la.apply_aperture(` call, which is text;
`test_audit_misc.py::test_d5_propagate_through_system_threads_dy_aperture` reads
only `result.dx` / `result.dy`; `test_v5_21_2_subsystem_audits.py` pins the
inverted-annulus refusal; the dispatcher-pin and walker files read SOURCE and
pin dispatch shape, not values.

One is worth naming because the change made it stronger rather than weaker:
`test_audit_misc.py::test_jax_aperture_matches_numpy` compares the two backends
at a 5 %-of-pixels bar; after section 2.4 the two are byte-identical, so the
test now passes with the whole margin.  Its loose bar is NOT tightened here (it
is B1-1's own fp32 contract and tightening it is not this work package's call),
but the exact cross-backend claim is asserted in
`tests/unit/test_c1_gray_edge_default.py::test_c1_the_system_chain_takes_the_same_default_on_both_backends`,
over both JAX routes and both arms.

### 3.4 The wider sweep: four more reds, three of them WP-C1's own

The 33-file blast-radius run is not the whole gate.  The census / walker /
dispatcher-pin / public-API / doc-consistency sweep (61 files, 2764 ids) raised
four more.  Each was premise-gated against the parent commit before being
called a move.

| red | whose | what was done |
|---|---|---|
| `test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached` | **WP-C1's** | 19 shipped lines named `v5.49.0` while `__version__` is `5.48.1`.  The gate's own reasoning is the right one and it was MEASURED, not assumed: over the last five releases, mentions of the version being released inside `lumenairy/**` at the release commit's PARENT were 0, 0, 0, 1 (a false positive) and 0, and the release commit touches exactly ONE library file to stamp the number.  Every version token is gone from `lumenairy/**`; the docstrings say what the default IS and point at the Migration note for when it moved. |
| `test_audit2609_a17_history_lint.py::test_no_module_accumulates_more_version_history` | **WP-C1's** | The same edits grew the version-history NARRATIVE in four modules (`apertures.py` 0 -> 1, `elements.py` 15 -> 17, `polarization.py` 38 -> 39, `system.py` 25 -> 35).  The A17 ratchet allows a module's narrative to shrink or stay, never grow -- the narrative belongs in the CHANGELOG, the Migration-Guide and `docs/history/`.  Same remedy, same commit.  Both gates green, and the baseline was NOT re-recorded (the counts are back at or below their recorded values). |
| `test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines` | **WP-C1's** | Six `file:line` citations in the `[5.47.0]` CHANGELOG block no longer named the line whose CONTENT they named, because this work package shifted `system.py` and `polarization.py`.  Re-anchored with the sanctioned, content-based, idempotent tool (`scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"`); only the six numbers changed. |
| `test_public_api.py::test_installed_metadata_version_matches_source_version` | **NOT WP-C1's** | The editable install in this box's site-packages carries `5.47.0` metadata while the source says `5.48.1`.  PREMISE-GATED: the same id fails on a clean `git archive 49ddf4bd` extraction on this box, so it is a box state (a stale `pip install -e .`), not a move.  Left alone; recorded here and in section 6. |

### 3.5 The validation suite: two throughput claims were measuring the wrong moment

`validation/elements/` is not in the unit sweep but it is a gate
(`validation/run_all.py`, wrapped by `tests/integration/`).  It stayed 31/31
green through the flip, and two of its green assertions were wrong anyway --
one of them within 0.5 % of its own bar.

The aperture is an **amplitude** mask.  The LINEAR sum of the mask is the
transmitted AREA; the QUADRATIC sum is the transmitted POWER of a
unit-amplitude field.  For a binary rim the two coincide, because `f` is 0 or
1.  For an area-averaged rim they differ by exactly the rim's own
`sum(f - f^2) dx^2`, which is bounded by `n_rim dx^2 / 4` (the maximum of
`f - f^2` is 1/4, at `f = 1/2`) and falls like the perimeter, i.e. as 1/N.
That is not a defect: the area-averaged AMPLITUDE is the correct band-limited
representation of the field just behind the stop -- it is what delivers the
second-order convergence in section 1 -- and it is precisely because it is the
right FIELD that it is not the right POWER.

Measured 2026-09-20, identical on both builds (a mask sum is an integer count
over 16, with no BLAS in it):

| fixture | area, hard | area, gray | power, hard | power, gray | rim px | bound `n_rim dx^2/4` |
|---|---|---|---|---|---|---|
| circular, N = 256, dx = 8 um, D = 1 mm | +0.0746 % | **+0.0074 %** | +0.0746 % | -0.4893 % | 364 | 0.7415 % |
| rectangle, N = 256, dx = 4 um, 200 x 150 um | +0.6400 % | **+0.0000 %** | +0.6400 % | -1.9900 % | 176 | 2.3467 % |

and the power deficit halves with the pitch: 0.4967 % / 0.2585 % / 0.1297 % at
N = 256 / 512 / 1024 on the circular fixture.

So the AREA went 10x better on the circle and became EXACT on the rectangle
(its rims sit at 25.00 and 18.75 pixels and a 4x4 lattice resolves a
quarter-pixel rim exactly), while the POWER reading the two assertions were
actually taking went to 1.99 % against a 2 % bar -- the S4 floor-bar shape
`docs/TESTING_STANDARDS.md` forbids, green and meaningless.  Both now assert
the AREA against a derived 5e-4 bar (the readings are 7.4e-5 and 0.0; the
pixel-centre rim they must beat reads 7.46e-4 and 6.4e-3, so the bar sits
between them with 1.5x and 12.8x) and the POWER inside the DERIVED two-sided
band `[area - n_rim dx^2/4, area]`, which is a bound, not a tolerance.
A third, `t_circular_aperture`, decided its "inside" set from pixel CENTRES and
read a mean amplitude of 0.9960 against a 0.99 bar; it now decides all three
sets from the pixel CORNERS and asserts the two outer ones EXACTLY.  31/31.

---

---

## 4. Byte identity, archive to archive

`validation/probe_c1_gray_edge/probe_fixtures.py` builds 15 APERTURE-touching
fixtures (one per in-library path that reaches `elements.apply_aperture`,
including the JAX and CuPy arms of the same xp-parametrised body and both JAX
chain routes) and 16 NON-aperture fixtures, and records the SHA-256 of
`np.ascontiguousarray(arr).tobytes()` with dtype and shape.
`compare_fixtures.py` compares a run against the `git archive 49ddf4bd`
extraction with a run against the worktree.  Both sides of every comparison are
the SAME build; nothing is compared across builds.

### 4.1 Claim A -- the way back is exact

Base tree with NO keyword vs new tree with `edge='hard'` (for the four entry
points that expose no `edge`, the new side runs that entry point's DOCUMENTED
way back):

| build | aperture fixtures byte-identical | moved | non-aperture byte-identical | moved |
|---|---|---|---|---|
| Windows py3.14 | **14 / 15** | 1 | **16 / 16** | 0 |
| WSL py3.12 | **13 / 14** | 1 | **16 / 16** | 0 |

(WSL is one aperture fixture short because CuPy is absent there; the probe
records that in its `unavailable` list rather than dropping it silently.)

The single moved fixture on both builds is `system_aperture_element_jax_eager`
-- the JAX slow path -- and it is section 2.4's pre-existing signed-zero defect,
not a consequence of the flip.  Measured precisely: at `49ddf4bd` that path's
digest (`93574601eb5b...`) already differed from its OWN jit'd kernel and from
the NumPy chain (both `387bcc150806...`) on the same input; after this change it
reads `387bcc150806...`, i.e. it is byte-identical to the parent commit's NumPy
and jit answers.  It moved from disagreeing with itself to agreeing.

### 4.2 Claim B -- the flip's blast radius stops at the aperture

Base tree vs new tree, both with NO keyword:

| build | aperture fixtures moved | non-aperture moved |
|---|---|---|
| Windows py3.14 | **15 / 15** | **0 / 16** |
| WSL py3.12 | **14 / 14** | **0 / 16** |

Every aperture-touching fixture moves -- that is the default move -- and no
non-aperture fixture moves at all.  The 16 that do not move:
`apply_gaussian_aperture`, `apply_apodized_pupil`, `apply_zernike_aberration`,
`apply_thin_lens`, `apply_mirror` (with its own `aperture_diameter` mask),
`apply_axicon`, `angular_spectrum_propagate`,
`rayleigh_sommerfeld_propagate(kernel='transfer')`,
`rayleigh_sommerfeld_propagate(kernel='spatial')` on a Gaussian,
`propagate_huygens_fresnel_freespace`, `thin_grating_efficiency_1d`,
`rcwa_efficiency_1d`, `pmm_efficiency_1d`, `apply_real_lens(surface_model='thin')`,
`zernike_basis_matrix`, `generate_turbulence_screen`.

The 15 that move: `ap_circular`, `ap_annular`, `ap_rectangular`,
`ap_decentred_anamorphic`, `ap_complex64`, `lyot_stop`,
`algebra_aperture_operator`, `jones_field_apply_aperture`,
`system_aperture_element`, `system_aperture_element_jax_jit`,
`system_aperture_element_jax_eager`, `rs_spatial_of_apertured_field`,
`hf_quadrature_of_apertured_field`, `jax_apply_aperture` (Windows and WSL),
`cupy_apply_aperture` (Windows only).

### 4.3 The GPU and JAX arms take the same default -- measured, not inferred

`apply_aperture` has no separate GPU or JAX implementation: one body dispatched
through `backend.array_namespace`, so the arms cannot diverge by construction.
That is a claim about the code, so it was measured about the ANSWER instead: on
Windows the JAX arm and the CuPy arm both move with the default (`8869b07d6f92`
-> `3c7496cc7eaf`, the same pair for both, and the same pair the NumPy arm would
give at that shape), and both return to the parent commit's bytes under
`edge='hard'`.  CuPy is present on the Windows box and absent under WSL; the
probe says so in its output rather than skipping.

---

## 5. Tests added

`tests/unit/test_c1_gray_edge_default.py`, 17 tests, none `slow`, whole file
10.6 s (slowest single test 2.9 s -- the brief's 60 s ceiling is not approached).

* the default is `'gray'` read from the SIGNATURE and from a CALL (a signature
  can say one thing while the body re-binds, and a body can behave greyly while
  `help()` still says `'hard'`), the call compared bit for bit;
* `edge_samples` is 4 AND 4 is the knee, re-measured on the ladder of 1.4, both
  sides;
* the way back is the binary mask bit for bit on all three shapes, and again
  under a decentred, anamorphic, complex64 call;
* the convergence DECISION on both kernels (1.3's table), plus the separate
  qualitative claim that the hard arm's defect is the ABSENCE of a rate, not a
  larger constant;
* the rim fractions: a rim laid exactly on a pixel centre reads exactly 0.5 and
  a corner exactly 0.25 (exact arithmetic on dyadic rationals, no bar); the 2-D
  mask equals the outer product of two independently built 1-D sub-sample counts
  exactly; fractions are bounded in [0, 1]; a blocked pixel is exactly zero for
  a NaN or +-inf input;
* both backends of the chain agree bit for bit on both arms, over the jit'd
  kernel and the slow path;
* **the mutation matrix**, three rows, each naming the test that catches it:

| mutation | caught by |
|---|---|
| the default silently reverts to `'hard'` | `test_c1_the_default_edge_is_gray_from_the_signature_and_from_a_call` |
| `edge_samples` moves off the knee (1, 2, 8 or 16) | `test_c1_edge_samples_default_is_four_and_four_is_the_measured_knee` |
| the grey mask's boundary fraction is wrong (sub-sample lattice anchored on the cell CORNERS instead of centred on the pixel) | `test_c1_a_rim_through_a_pixel_centre_reads_exactly_one_half` (the mutant reads 0.75 on the +x rim, 0.5 on the -x rim and 0.5625 in the corner, so it fails the half, the quarter and the mirror symmetry) and, independently, the separable-area identity |

Each row runs the named test's own check function against the mutant and asserts
that it FAILS; a row that passed under mutation would mean the named test is not
load-bearing.

---

## 6. What could not be measured

* **CuPy under WSL.**  Absent there.  The CuPy arm is measured on Windows only,
  and the probe records the absence in `unavailable` instead of skipping.
* **`propagate_through_system_jax`'s slow path on a real GPU.**  Measured on the
  CPU backend of JAX only; the signed-zero finding of 2.4 is an XLA rewrite
  difference between jit and eager and is not expected to be device-specific,
  but that is reasoning, not a measurement.
* **The generated-script path end to end.**  `io/codegen.py` emits an
  unkeyworded `la.apply_aperture(...)` call; that the emitted TEXT is unchanged
  is pinned by `test_audit_io.py`, but no probe here executes a generated script
  against both trees.
* **Whether the 5 %-of-pixels cross-backend bar in
  `test_jax_aperture_matches_numpy` should be tightened.**  It is now satisfied
  with the whole margin, and the exact claim is asserted in the new file, but
  changing B1-1's own bar is a separate decision.
* **`Migration-Guide.md` has no 5.48.0 section** although 5.48.0 moved two
  answers (the FGA and GBD reference-plane repairs, both with Migration
  paragraphs in the CHANGELOG).  Noticed while adding the 5.49.0 section; not
  fixed here, since back-filling another release's guide entry is not this work
  package's scope.
* **`.test_durations`** was validated as JSON (16 596 entries) and left
  untouched; the staleness gate
  (`test_audit2609_a15a_durations_staleness.py`, 4 passed) does not require an
  entry for a newly added file.
* **`test_public_api.py::test_installed_metadata_version_matches_source_version`
  cannot pass on this box** and does not pass at the parent commit either: the
  editable install's metadata reads `5.47.0` against a `5.48.1` source.  The
  remedy is `pip install -e .` on the box, which is not this work package's to
  do (it would change what every other session on this machine imports).
* **Whether the full unit suite is green.**  What was run is the 33-file
  grep-selected aperture set (1394 ids), the 61-file sweep (2764 ids) and the
  validation `elements` file; the whole 14 666-id suite was not, and a release
  gate would need it.
* **The `edge_samples` knee on the HF quadrature.**  It is pinned on the RS
  ladder, where WP-B11 measured it.  On the HF quadrature at N = 256 the
  readings are non-monotone in `edge_samples` (2 -> 3.4183e-03 is marginally
  better than 4 -> 3.4928e-03), so the knee statement is kernel-specific and is
  asserted only where it was measured.

# VERIFY-A9 — independent re-verification of WP-A9 (designer UI, U1–U7)

Branch `audit-fixes-2026-09`.  Diff under test: `58ad836c` ("fix(ui): WP-A9 …"),
base `58ad836c^` — 35 files, +3707 / −300.  PySide6 is not installed on this
workstation; every number below was measured through the auditor's Qt stub
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub`), which is
also the harness for the new tests.  Nothing skips on PySide6's absence.

I did not write the fixes.  Every oracle below was built inside the verification
scripts or the new test file — a hand-written Welford ABCD product, a
hand-written paraxial marginal-ray trace, hand-built unfolded surface lists, and
a re-implementation of the pre-fix semi-diameter rule — so no assertion is
checked against the code under test.

---

## 1. Verdicts

| ID | verdict | what I measured |
|---|---|---|
| **U1** | **VERIFIED** | Exported-vs-layout ABCD agrees to the last bit on **four fold topologies WP-A9 did not use** (two mirrors; a curved 45° fold + coord break; a mirror as the LAST surface; `is_stop` on the mirror), and all of them agree with an independent hand-computed Welford matrix.  Unsigned axial path conserved on all four.  Round-trips through `load_prescription` and through the user lens library. |
| **U2** | **VERIFIED** (was "partially fixed") — I implemented the granted `trace.py` finish | Mirror semi-diameter **0.006 → 0.025 m** on the audit fixture; **[8, 8, 8, 8] → [25, 20, 8, 8] mm** on a two-mirror design.  Bit-identical on 58 of the repository's 62 reachable prescriptions; the 4 that move are the designer's folded exports.  Two companion changes are required outside my ownership (§5.1, §5.2). |
| **U3** | **VERIFIED** | Ring radii and azimuths match the closed form to ≤ 2.3e-16 relative over six (`num_rings`, `rays_per_ring`) pairs the WP did not use, `num_rings·rays_per_ring + 1` distinct directions each time.  Object-distance edges (negative first gap, zero first gap, 1 µm conjugate, infinite/zero/negative form field, six non-point source kinds) all behave. |
| **U4** | **VERIFIED** | Globals restored on every exit path of the **real worker** — success, `requestInterruption()`, an exception in `_run_impl`, an exception whose `__str__` itself raises — and of the context manager under `KeyboardInterrupt` (a `BaseException`, which a `try/finally` bug would leak), under nesting, and over a pre-existing user `set_max_ram(32)`.  Combo index 0 touches nothing. |
| **U5** | **VERIFIED-WITH-NOTES** — **one regression found and fixed** (§4.1) | The router, the `lens_model_used` reporting and the both-ways fold detection are right, including a NON-fold refusal (a bad per-function kwarg is reported verbatim).  Axial path conserved for 1–4 mirrors.  **But** the new gap-carry put a dropped mirror's air gap on the wrong surface and in the wrong medium — EFL/BFL 68.236 / 21.085 mm against a hand-built unfolded truth's 82.515 / 24.842 mm.  Fixed. |
| **U6a–i** | **VERIFIED** (U6c with a note) | See §3.  `_on_finished` survives six `N`/`N_out` combinations including `N_out = 3` and a legacy payload with no `N_out`; all six Insert ▸ Source presets work and checkpoint; λ and polarization survive a form edit; the model wavelength reaches the source on all three paths **and after an undo**; emitter counts reject 0 / −1 / 3.5 / `'x'` / `None` and accept numpy scalars; world and local surface lists agree exactly on a three-element system with and without a coord break; the PSF pupil matches an independent paraxial marginal-ray height to 0.7 % on a two-element system where it is **not** EPD/2; the optimizer clone survives NaN / ±1e30 merit probes without touching the live model; three real workers honour `requestInterruption()` at runtime. |
| **U7** | **VERIFIED-WITH-NOTES** — **two defects found and fixed** (§4.2, §4.3) | 10 of the 19 items spot-checked against independent oracles, all correct (§3.7).  Structural sweep widened from the WP's 14 modules to **all 49**: 20 `QThread` subclasses, 0 shadowing `finished`, 0 without a cancellation path.  **But** `GlobalSearchWorker.run()` raised on its first restart (pre-existing, on a line the WP edited) and the absolute-coordinates Distance column moved folded elements (introduced by the WP). |

Nothing in the WP's report was found to be numerically wrong.  Every "after"
number it quotes reproduces (§6).

---

## 2. Defects I found and fixed

All four are in files WP-A9 owns (`lumenairy/ui/`) plus the `raytrace/trace.py`
block granted to me.  Each has its own fail-before demonstration (§7).

### 2.1 `_filter_wave_optics_surfaces` carried the fold's gap forward, not backward  [P1, REGRESSION introduced by `58ad836c`]

`lumenairy/ui/waveoptics_dock.py:390–418`.

A `Surface.thickness` is the gap **after** that surface, propagated in the
medium **after** it (`_run_impl`'s per-surface loop: phase screen → aperture →
`angular_spectrum_propagate(E, ts.thickness, wv / n2, …)`).  Removing surface
*k* therefore has to merge the two legs that met at it onto surface *k−1*.  The
WP's new carry added it to surface *k+1* — one gap too late, and in that
surface's exit medium.  Its own comment describes the correct algorithm ("added
to the previous KEPT surface's thickness … a carry with no previous kept surface
is dropped"); the code did the opposite, and the WP's test only asserted the
*total*, which the wrong placement also conserves.

Measured on singlet → 25 mm air → flat fold → 40 mm air → singlet:

| | thicknesses (m) | EFL (m) | BFL (m) |
|---|---|---|---|
| shipped (forward carry) | `[0.003, 0.025, 0.043, 0.0]` | 0.06823621185021057 | 0.021085332140327456 |
| **fixed** (backward carry) | `[0.003, 0.065, 0.003, 0.0]` | **0.08251493722378367** | **0.02484249389527306** |
| hand-built unfolded truth | `[0.003, 0.065, 0.003, 0.0]` | 0.08251493722378367 | 0.02484249389527306 |

17.3 % EFL / 15.1 % BFL, and 40 mm of air was being propagated as N-BK7
(λ/1.5168).  The fixed version is bit-identical to the hand-built truth.  A
carry with no preceding kept surface is now dropped (the field is constructed AT
the first surface, so a leading gap is not propagated — the same rule the
source-to-first-surface gap already follows); a carry left over at the end is
flushed onto the last kept surface.  Verified on four fold topologies
(mirror first / middle ×2 / 45° fold riding a coord break) against hand-built
lists; the WP's own `test_u5_unfolding_preserves_the_axial_path` still passes.

Also fixed alongside it: `_prescription_from_surfaces` (the helper that builds
the unfolded-equivalent prescription for the router) `continue`d over coord-break
Surfaces and dropped their transfer thickness — the U1 defect one level down.
Latent rather than live today (a cb carrying a gap is always adjacent to a
mirror and therefore already dropped upstream), but it is the exact shape the
audit catalogues, so it now carries the gap into the preceding gap.

### 2.2 `GlobalSearchWorker.run()` raised before it emitted anything  [P1, pre-existing; the WP edited the line]

`lumenairy/ui/optimizer_dock.py:1558–1600`.

```
ValueError: too many values to unpack (expected 2, got 3)
```

`opt_variables` entries are `(elem_idx, surf_idx, field)` triples; the restart
loop unpacked them into `(row_idx, col_idx)`.  This fires on the FIRST restart,
before any `finished_result` emission, so the Optimizer dock's Global-search
button left the UI disabled and the log stuck at "Running…" until restart.
Present before `58ad836c` (`git show 58ad836c^:lumenairy/ui/optimizer_dock.py`
has the same unpack against `self.model.opt_variables`); WP-A9 changed that line
to `live_opt_variables()` without noticing.  The conic test in the same loop
keyed off `col_idx == 7`, a column index from a table model this code no longer
uses.

Fixed: unpack the triple, test `field == 'conic'` (with a why-comment for the
additive-vs-multiplicative perturbation), and wrap `run()` so exactly one
`finished_result` is emitted on every path.  Verified: completes for a radius
variable, a conic variable and both together; the live model is bit-unchanged;
`requestInterruption()` still yields `(False, 'Cancelled -- best so far: …')`
with `apply_result_on_failure` set; a merit that always raises yields exactly
one `(False, 'Global search failed: RuntimeError: merit exploded')`.

### 2.3 The absolute-coordinates Distance column moved folded elements  [P2, REGRESSION introduced by `58ad836c`]

`lumenairy/ui/model.py` `set_display_distance` + the new
`_display_distance_slope`.

U7 changed `element_z_positions_mm()` from a cumulative sum of `distance_mm` to
`Element.origin[2]` — correct, and it fixes the unfolded round trip the WP
measured (40.0 → 37.0 mm per write-back on a 3 mm singlet).  But the column is
now a **world Z** while the value written back is a distance along the **optical
axis**, and `set_display_distance` still computed `value − prev_back_z`.  The two
coincide only while the axis is parallel to world Z:

| mirror fold | shown (mm) | distance before → after (shipped) |
|---|---|---|
| 0° (control) | 90.0 | 40.0 → 40.0 |
| 10° | 87.588 | 40.0 → **38.168** |
| 30° | 70.000 | 40.0 → **23.094** |
| 45° | 50.000 | 40.0 → **0.0** |

Pre-WP-A9 this round-tripped on folded systems (both sides used the axial
cumulative sum) and was wrong on unfolded ones, so the WP traded one for the
other.

Fixed: world Z is affine in `distance_mm` with slope `R[2, 2]` of the axis the
advance runs along (`recompute_element_frames` applies `origin += d * R[:, 2]`
exactly once), so the exact inverse is one step from where the element is now.
The new `_display_distance_slope` picks the right `R` using the cb_pre / cb_post
asymmetry `recompute_element_frames` documents.  When the slope is zero — a 90°
fold, where the column cannot describe the distance at all — the write is
declined and no undo checkpoint is taken, instead of silently moving the element
to 0.

Verified exact (≤ 1e-9 mm) for mirror folds of 0/10/20/30/45/60°, with and
without a decenter on the tilted element, for a cb_pre tilt after a lens, and
**bit-identical** (`==`, not `approx`) on unfolded systems.

### 2.4 `raytrace/trace.py` — the U2 finish (granted by the orchestrator)

`lumenairy/raytrace/trace.py:534–586`.  The block read the per-surface
`'semi_diameter'` key and then unconditionally `min()`-ed it against
`prescription['elements']`, matched by index **within the refracting entries**.

I implemented the `elif` the orchestrator asked for.  I did **not** implement
"match positionally over the whole `elements` list" unconditionally: that is
measurably wrong for the `.zmx` / CodeV shape, where `surfaces` is the lens-only
list and `elements` includes the mirrors, and it breaks
`test_audit_w5_raytrace_bundles::test_resolver_matches_numpy_invalid_and_missing`
(an `elements` list with a non-`'surface'` entry and the same length as
`surfaces`).  Instead the fallback picks its index from the shape:

```python
chronological = (len(elements) == len(p_surfs)
                 and any(_s.get('is_mirror') for _s in p_surfs))
```

A mirror marked on a `surfaces` entry is exactly the condition under which the
refracting-only filter mis-indexes, and the equal-length requirement keeps the
positional read off `elements` lists with non-optical entries.  `None` in an
`elements` entry's `semi_diameter` is skipped rather than raising `TypeError` on
`> 0`.  The `min()` against the `aperture_diameter / 2` default is kept in the
fallback branch — that is the documented semantics and removing it would change
every loader.

**Measured effect.**

| fixture | before | after | correct |
|---|---|---|---|
| audit fold, mirror | 0.006 | **0.025** | 0.025 |
| audit fold, lens S1 / S2 | 0.006 / 0.006 | 0.006 / 0.006 | ✔ |
| two mirrors + singlet | [0.008, 0.008, 0.008, 0.008] | **[0.025, 0.020, 0.008, 0.008]** | ✔ |
| pre-WP-A9 library entry (no per-surface key) | 0.006 (the next lens's) | **0.0127** (its own 25 mm, capped by aperture/2) | — |

**Scope, measured.**  62 prescriptions — `make_singlet` / `make_doublet` /
`make_cylindrical` / `make_biconic`, 15 `.zmx` fixtures (one with a mirror),
seven `.seq` fixtures (one with two mirrors), three designer exports, and
`normalize_prescription` of each — resolved under the shipped rule and the new
one: **58 bit-identical, 4 changed, and all four are designer folded exports.**
Collateral: 153 passed in the raytrace test files, 106 in io / CodeV / lens.

---

## 3. Per finding — what I measured

### 3.1 U1 — VERIFIED

Four fold topologies the WP did not use.  Three independent paths compared:
`system_abcd(build_trace_surfaces())` (the layout), `system_abcd(
surfaces_from_prescription(to_prescription()))` (what the 18 dock call sites
see), and a Welford matrix product written in the verification script.

| fixture | EFL (m) | BFL (m) | exported == layout == hand |
|---|---|---|---|
| two mirrors + singlet | 0.2225221628768968 | 0.0744807042287246 | ✔ bit-identical |
| curved 45° fold + tilted biconvex | −0.07423229646679669 | −0.10366598146440285 | ✔ |
| singlet then mirror LAST | 0.08747238214182079 | 0.06382000508037293 | ✔ |
| stop on the mirror | 0.17877415918110384 | 0.12780222366203894 | ✔ |

Unsigned axial path conserved on all four (0.144 / 0.043 / 0.050 / 0.034 m).
`is_stop` on a mirror survives export → `surfaces_from_prescription` →
`load_prescription` (it lands back on the Mirror element, not a lens).
Unfolded singlet export still key-for-key equal to `make_singlet`, EFL
0.09928851726861038 / BFL 0.09729328309216069 from both.
User-library round trip (`user_library.save_lens` / `load_lens`): `is_mirror`,
`semi_diameter`, `is_stop`, `stop_index`, `object_distance`, `image_distance`
all survive; EFL/BFL identical to the last bit before and after; `inf`
semi-diameters round-trip as `inf`; re-loading through `load_prescription` gives
back `['Source', 'Mirror', 'Singlet', 'Detector']` with semi-diameters
`[[25.0], [6.0, 6.0]]` and the same layout ABCD.

Degenerate cases: a Source-only model exports `surfaces: []`, which
`validate_prescription` rejects with a named error (no silent garbage); a
single-mirror model exports one surface, zero thicknesses and validates.

### 3.2 U2 — VERIFIED (see §2.4 for the change and its scope)

Both `elements` shapes now index correctly without a per-surface key:
lens-only + mirror-in-elements → `[0.006, 0.006]` (the refracting filter, still
right); chronological → the mirror consults its own entry.  The pre-fix rule,
replayed in-test, gives `[0.006, 0.006]` and `[0.006, 0.006, 0.0127]`.

### 3.3 U3 — VERIFIED

Six sampling densities the WP did not use, on a 200 mm conjugate:

| `num_rings`×`rays_per_ring` | rays | distinct | max \|ρ/ρ_exact − 1\| | max \|Δazimuth\| |
|---|---|---|---|---|
| 1×1 | 2 | 2 | 0.0 | 0.0 |
| 1×3 | 4 | 4 | 2.2e-16 | 8.9e-16 |
| 2×5 | 11 | 11 | 0.0 | 4.4e-16 |
| 4×12 | 49 | 49 | 2.2e-16 | 8.9e-16 |
| 5×1 | 6 | 6 | 0.0 | 0.0 |
| 6×7 | 43 | 43 | 2.2e-16 | 4.4e-16 |

Chief ray on axis in every case; `N` finite for every ray.

Object distance from the geometry: first optic at 100 mm with the form field at
1000 mm → 0.100 m (the pupil fills 10× more than the stale field would give);
first optic at 0 mm or at a Zemax-signed −30 mm → falls back to the form field
rather than dividing by zero; a 1 µm conjugate gives ρ_max = 12700 and still a
finite on-shell bundle (`_make_bundle` normalises); all six non-point source
kinds report 0.0 = "object at infinity" and export it.

### 3.4 U4 — VERIFIED

`(USE_PYFFTW, USE_SCIPY_FFT, get_max_ram())` read before and after.  Library
default here is `(True, True, None)`.

* Backends `'default'`, `'numpy'`, `'scipy'`, `'pyfftw'`, `'cupy'`, `''`, `None`
  — restored in every case; `'default'`, `''` and `None` touch nothing inside.
* Memory limits `None`, `0`, `4`, `8.5`, `64·2³⁰` — inside values correct
  (`4 → 4294967296`, `8.5 → 9126805504`), restored in every case.
* Exception, `KeyboardInterrupt` (a `BaseException` — the classic
  generator-`finally` leak), nesting (inner restores to the outer's state, outer
  to the library's), and a pre-existing `set_max_ram(32)` preserved.
* The real `WaveOpticsWorker.run()`: success, `requestInterruption()`, an
  injected `RuntimeError`, and an exception whose `__str__` raises — all restore
  the globals, all emit exactly one payload, nothing escapes `run()`.
* `N_out` / `dx_out` present on the success payload.

### 3.5 U5 — VERIFIED-WITH-NOTES (defect in §2.1)

| unfold | lens model | `lens_model_used` | reason |
|---|---|---|---|
| on | `real_lens` | `real_lens (unfolded equivalent)` | — |
| on | `real_lens_traced` | `real_lens_traced (unfolded equivalent)` | — |
| off | `real_lens` | `asm (fallback)` | `ValueError: apply_real_lens: prescription ha…` |
| off | `real_lens_traced` | `asm (fallback)` | `ValueError: apply_real_lens_traced: prescrip…` |
| either | `asm` | `asm` | — |

`apply_real_lens` refusing for a **non-fold** reason is reported verbatim:
`asm (fallback)` / `TypeError: apply_real_lens() got an unexpected keyword
argument 'not_a_kwarg'`.  `_prescription_has_mirror` detects an explicit
`is_mirror`, a legacy `element_type: 'mirror'`, and neither, and survives `None`
and a non-dict.  Axial path conserved for 1, 2, 3 and 4 mirrors (0.034 / 0.074 /
0.114 / 0.154 m), no mirror left in the unfolded list in any case.

### 3.6 U6a–i — VERIFIED

* **U6a** — `_on_finished` driven through the real method at `(N, N_out)` =
  (256, 64), (512, 128), (128, 128), (1024, 37), (64, 8), (256, 3): no
  `IndexError`, and the imshow extent equals `max(1, N_out//8) · dx_out · 1e6`
  in every case.  A legacy payload carrying no `N_out` / `dx_out` also works.
* **U6b** — all six presets install the source, sync λ (632.8 nm kept) and add
  exactly one undo checkpoint each.  *Note:* an unknown kind string is installed
  verbatim (`source_type='not_a_kind'`); only the six wired menu actions reach
  it, so this is a robustness nit, not a live defect (§8.1).
* **U6c** — after a form edit: λ 780 nm kept, polarization `'rcp'` kept,
  `emitter_nx` an `int`.  A non-numeric entry is ignored and the source
  survives.  *Note in §8.2:* the form's **other 11** numeric fields are written
  back from the widget text, which `_load_source_ui` keeps in sync when the
  Source row is selected — I could not exercise the refresh headlessly.
* **U6d** — model λ reaches the source through `set_wavelength`, `set_source`
  and `load_prescription`, **and after an undo** (both roll back to 1310.0
  together).
* **U6e** — `emitter_nx` ∈ {0, −1, 3.5, `'x'`, `None`} all raise `ValueError`
  with the §2 prefix; `np.int64(6)` / `np.float64(4.0)` are accepted and
  `to_source` returns a (32, 32) field.  Every `emitter_n*` read in the package
  is guarded (`model` coerces, `layout_2d`/`layout_3d` wrap in `int()`,
  `element_table` has `_SRC_INT_FIELDS`).
* **U6f** — three elements, with and without a 5° tilt: world and local
  thicknesses identical (`[0.003, 0.04, 0.003, 0.025, 0.002, 0.0]`),
  `find_paraxial_focus` 0.012880703023804329 from both, `system_abcd` EFL
  0.0686197011201607 from both.
* **U6g** — on a two-element system the exit-pupil ray height is **5.817e-03 m**
  against an independently traced paraxial marginal height of 5.859e-03 m
  (0.71 %, the fixture's own spherical aberration) and **not** EPD/2 = 12.7 mm;
  34× the image-plane radius; every alive ray on one plane.  The singlet arm:
  1.258e-02 vs 1.244e-02 (1.10 %), 50× the image radius.
* **U6h** — after merit probes at NaN, ±1e30, 0 and 1.5·R₀ the live
  `SurfaceRow.radius` and `Element.origin` are bit-unchanged; the clone shares
  no element, surface **or source** object with the live model.
* **U6i** — runtime, not just structural: `_PolyStrehlWorker` →
  `{'success': False, 'error': 'Stopped by user'}`; `_CalibrationWorker` →
  `(12.0, 'interrupted')`; `OptimizeWorker` stops after 2 of 200 iterations and
  its `ThreadCancellableProgress` reads `should_stop is True`.

### 3.7 U7 — 10 of 19 spot-checked

| item | oracle | result |
|---|---|---|
| `finished` shadowing | sweep over **all 49** ui modules | 20 workers, **0** shadowing, **0** with no cancellation path (the WP swept 14 modules) |
| Start at / End at | the span map vs the run's own reported range | `{1:(0,2), 2:(2,4), 3:(4,6)}`; `(1,1)` → `elements 1..1 (surfaces 0..1)`, `(2,3)` → `2..3 (2..5)`, `(1,2)` → `1..2 (0..3)`; an inverted range reports `empty element range` |
| `power_in` plane | the same run with and without `'Source'` in the save list | 6.5536e-06 and 92.723 % throughput in all three configurations |
| explicit `image_distance` | `last.world_origin + bfl · last.world_R[:,2]` | exact at detector 0 / 100 / 250 mm; the default still prefers the detector when it exists and falls back to the paraxial focus when it is at 0 |
| `_apply_real_lens_asm_equiv` | its own docstring's calibration points | n=2 → 1.1, n=3 → 2.2 |
| absolute-Z round trip | `Element.origin[2]` | exact unfolded; **defect on folds, fixed — §2.3** |
| `min_thickness` merit | `(1 − t)²` by hand | 0.0 compliant; 0.5625 for one 0.25 mm leg; 0.8125 for two; **a cemented doublet's 0.4 mm cement leg is counted** (0.36) |
| `diagnostics` separator | source read | `os.path.basename`, no `chr(92)` |
| designer int wavelength | the guard replayed over 6 inputs | 1.31e-6 and 1e-6 converted; 0, 1310, `True`, 1550.0 not |
| `bulk_edit` | undo depth | 3 nested mutators → 1 step; `_suppress_depth` back to 0 after an exception inside the block; a later single edit still checkpoints |

---

## 4. Tests I added

`tests/unit/test_audit2609_a9_verify_ui.py` — **34 tests**, on the same Qt stub,
reusing `test_audit2609_a9_ui`'s install/park bootstrap so there is exactly one
owner of the discipline that keeps the sibling files' skip set intact.

| test | pins |
|---|---|
| `test_u1_folded_export_matches_an_independent_abcd[4 fixtures]` | exported == layout == hand-computed Welford ABCD, plus the mirror flags and the axial path |
| `test_u1_is_stop_on_a_mirror_round_trips` | the stop can sit on the fold and survives export → trace → import |
| `test_u2_per_surface_semi_diameter_is_not_clipped_by_the_elements_matcher` | 0.025 / [0.025, 0.020, …], with the pre-fix rule replayed in-test |
| `test_u2_both_elements_shapes_index_correctly_without_a_per_surface_key` | both `elements` layouts |
| `test_u2_elements_fallback_is_bit_identical_for_every_library_producer` | bit-identity guard over ≥ 20 builders / loaders / normalizations |
| `test_u3_cone_matches_the_closed_form_at_several_densities[6]` | ρ and azimuth closed form, two-sided against the √2 pre-fix value |
| `test_u3_object_distance_edges` | geometry beats the form field; degenerate gaps; six source kinds |
| `test_u4_context_manager_restores_on_every_exit_including_baseexception` | `KeyboardInterrupt`, nesting, a pre-existing user cap |
| `test_u4_the_real_worker_restores_the_globals_on_every_exit_path` | four exit paths through `run()` |
| `test_u5_dropped_mirror_gap_lands_on_the_preceding_surface` | thicknesses, media and ABCD against a hand-built unfolded list |
| `test_u5_a_leading_fold_drops_its_gap_and_a_trailing_fold_keeps_it` | the two boundary cases |
| `test_u5_unfolded_equivalent_prescription_keeps_the_coord_break_gap` | the cb transfer thickness through the router helper |
| `test_u5_router_reports_a_non_fold_refusal_too` | the fallback reason is the real one |
| `test_u6g_exit_pupil_radius_matches_an_independent_paraxial_marginal_ray` | the pupil on a system where it is not EPD/2 |
| `test_u7_global_search_completes_and_emits_exactly_once[3]` | the unpack fix, exactly one emission, live model untouched |
| `test_u7_global_search_emits_once_even_when_the_merit_always_raises` | the guaranteed emission |
| `test_u7_absolute_z_column_round_trips_through_a_fold[6]` | 0/10/20/30/45/60°, including the degenerate no-op |
| `test_u7_absolute_z_column_is_bit_identical_on_unfolded_systems` | the unfolded control (`==`, not `approx`) |
| `test_u7_min_thickness_merit_counts_a_cemented_interface` | `surfaces[:-1]` still reaches internal surfaces |

Against `docs/TESTING_STANDARDS.md`: every numeric bar carries its derivation,
its error floor and the measured value in the docstring; the two real-ray-vs-
paraxial bars quote the aberration they allow for and the decades to the defect
they exclude; nothing reads a wall clock or asserts a speed-up; nothing calls
`pytest.skip` (the missing PySide6 is handled by the stub, not by skipping); all
fixtures are constructed through the public API.

**Fail-before**: 15 pinned properties replayed against the pre-fix code
in-process (`prefix_check_verify.py`, reproduced in §7) — the shipped
`_filter_wave_optics_surfaces` / `_prescription_from_surfaces`, the shipped
`GlobalSearchWorker.run`, the shipped `set_display_distance`, and the shipped
semi-diameter rule.  **13 fail; the remaining 2 are the deliberate unfolded
controls** (the 0° arm of the fold parametrization and the bit-identity guard),
which are supposed to pass on both sides.

I reviewed WP-A9's own `tests/unit/test_audit2609_a9_ui.py` (36 tests) against
the same standard.  It is sound; two notes, neither worth changing:
`test_u6g_*` asserts `r_pup ≈ epd/2` at `rel=0.02`, which is true for its
singlet fixture but is not the general property (my test uses the marginal-ray
oracle instead), and `test_u5_unfolding_preserves_the_axial_path` asserts only
the total, which is exactly why §2.1 slipped through — my
`test_u5_dropped_mirror_gap_lands_on_the_preceding_surface` pins the placement.

---

## 5. Requested changes outside my ownership

### 5.1 `lumenairy/raytrace/jax_trace.py` — the twin must follow  [P1, REQUIRED with §2.4]

`_resolve_semi_diameters` (≈ line 850) documents itself as mirroring
`trace.surfaces_from_prescription` "key for key".  It no longer does, and the
two backends now disagree — measured:

```
rx = surfaces [sd 2e-3, sd 1e-3], elements [sd 1e-3, sd 3e-3]
    numpy trace : [0.002, 0.001]          jax resolver: [0.001, 0.001]
UI folded export (mirror 25 mm + lens 6 mm)
    numpy trace : [0.025, 0.006, 0.006]   jax resolver: [0.006, 0.006, 0.006]
```

jax **is** importable on this machine, so this is live.  Exact replacement for
the loop body:

```python
    surfaces_raw = prescription.get('surfaces', [])
    ...
    for i, s in enumerate(surfaces_raw):
        sd = default_semi
        ps_sd = s.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        elif elements is not None:
            chronological = (len(elements) == len(surfaces_raw)
                             and any(_s.get('is_mirror')
                                     for _s in surfaces_raw))
            if chronological:
                match = elements[i] if i < len(elements) else None
            else:
                match = refr_elems[i] if i < len(refr_elems) else None
            if isinstance(match, dict):
                elem_sd = match.get('semi_diameter', np.inf)
                if (elem_sd is not None and np.isfinite(elem_sd)
                        and elem_sd > 0):
                    sd = min(sd, float(elem_sd))
        semi_ds.append(float(sd))
```

and its docstring's numbered rule 3 becomes "consulted only when the per-surface
key is absent".  Owner: WP-A1 / raytrace.

### 5.2 `tests/unit/test_audit_w5_raytrace_bundles.py` — one test pins the U2 defect  [P1, REQUIRED with §2.4]

`test_resolver_precedence_per_surface_then_elements_min` (line 205) asserts
`_resolve_semi_diameters(rx) == _numpy_sds(rx) == [1e-3, 1e-3]`, i.e. that a
**present** per-surface key is `min()`-ed down by `elements`.  That is precisely
the semantics U2 says is wrong.  It is the only failure left in the raytrace,
io, CodeV and lens test files (1 failed / 153 passed, and 1 / 106).  Requested
restatement:

```python
def test_resolver_precedence_per_surface_wins_over_elements():
    """Precedence parity (audit U2): a PRESENT per-surface key is the
    surface's own aperture and wins outright; the 'elements' matcher is a
    fallback for prescriptions that carry their apertures only there.
    Pre-U2 both backends returned [1e-3, 1e-3] -- the 'elements' entry
    min()-ed a stated 2 mm semi-aperture down to 1 mm with an index that
    is off by one per mirror."""
    ...
    assert _resolve_semi_diameters(rx) == _numpy_sds(rx) == [2e-3, 1e-3]
```

`test_resolver_matches_numpy_invalid_and_missing`,
`test_resolver_matches_numpy_elements_only` and
`test_resolver_no_aperture_keys_stays_open` pass unchanged.  Owner: WP-A1 /
raytrace.

### 5.3 `lumenairy/user_library.py` — the migration WP-A9 deferred is now half-done  [P2, optional]

WP-A9 §6.1 defers back-filling `is_mirror` / `semi_diameter` / `is_stop` onto
folded prescriptions saved before the fix.  §2.4 closes part of it for free: a
pre-fix library entry is the "chronological, no per-surface key" shape, so its
mirror now resolves to its own aperture (capped by `aperture_diameter / 2`)
instead of the following lens's — measured 0.006 → 0.0127 m on the audit
fixture.  What is still missing from such an entry is `is_mirror` itself, so its
ABCD is still the air→air no-op.  The deferred `load_lens` back-fill is still
worth doing; it is now smaller.

### 5.4 `tests/conftest.py` — WP-A9's §5.3 still stands

Lines 245–255 still say `lumenairy/ui/waveoptics_dock.py` "clears `USE_PYFFTW`
unconditionally".  It does not (U4, re-measured above).  Comment only.

### 5.5 WP-A9 §5.2 is already closed — no action

WP-A9 asked for `allow_unfolded_equivalent` to be honoured by the per-surface
mirror guard in `_lens_real.py`.  On the committed tree it already is (WP-A2's
`b97c0b6e`): measured, `apply_real_lens` refuses the folded export without the
flag and accepts it with the flag, and the refusal text now names the flag.  The
dock does not rely on it (it unfolds first), which is still the better path.

---

## 6. Repro scripts — quoted against the WP's after-numbers

All 11 re-run (`t11` needs the stub on `PYTHONPATH`; its own `sys.path` insert
is relative and only works from the repo root).

| script | measured now | WP-A9 claimed | verdict |
|---|---|---|---|
| `t1_singlet` | EFL 0.09928851726861038, BFL 0.09729328309216069, UI == `make_singlet` key for key | same | ✔ |
| `t2_mirror` | exported EFL 0.17877415918110384 / BFL 0.12780222366203894 == internal; `is_mirror`/`semi_diameter` present; **mirror sd 0.025** | same, except sd 0.006 (the residual §2.4 closes) | ✔ + improved |
| `t3_cb` | UI world frames match `world_surfaces_from_prescription` in orientation and relative position | unchanged | ✔ |
| `t4_pointsrc` | 25 rays, **25** unique (L, M), `L == M everywhere: False` | same | ✔ |
| `t5_wv` | model 632.8 / source 632.8; max \|Δφ\| **3.531376110613045e-17** rad | 3.53e-17 | ✔ |
| `t6_srcprop` | `source` still a read-only property (the U6b guard is still meaningful) | same | ✔ |
| `t7_emitter` | `emitter_nx` **int 12**, `to_source OK (64, 64)`; the λ/polarization lines are the script's own constructor call, unchanged by design | same | ✔ |
| `t8_docks` | `apply_real_lens` still refuses the fold by design; the refusal now names `allow_unfolded_equivalent` (WP-A2) | same shape | ✔ |
| `t9_psfpupil` | "31 of 65536" — the script re-implements the pre-fix binning, so it is unchanged by design; the dock path is measured by the tests | as stated | ✔ |
| `t10_abcd_cb` | world == local thicknesses; focus 0.04011208679014488 both; EFL 0.07297144166448877 both | same | ✔ |
| `t11` | local total 0.043, exported total 0.043, **LOST: 0.0** | same | ✔ |

---

## 7. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `tests/unit/test_audit2609_a9_ui.py` | **36 passed** | 5.5 s |
| `tests/unit/test_audit2609_a9_verify_ui.py` | **34 passed** | 1.8 s |
| the 12 sibling UI files, alone (control) | 47 passed, **38 skipped** | 10.1 s |
| the 12 + `a9_ui` | 83 passed, **38 skipped** | 15.8 s |
| the 12 + `a9_ui` + `a9_verify_ui` | 110 passed, **38 skipped** | 14.8 s |
| …the same, reversed collection order | 110 passed, **38 skipped** | 15.3 s |
| the 12 + both A9 files + 6 raytrace files | 268 passed, 38 skipped | 53.8 s |
| `test_audit_w3_raytrace_parity`, `w5_raytrace_bundles`, `w6_raytrace`, `niche_audit_w3_raytrace_sources`, `v5_4_6_wave8_raytrace`, `v5_4_retrace_ghost_path`, `niche_audit_a1_radial_metrics`, `v5_4_2_run_trace_empty_prescription` | 153 passed, 2 skipped, **1 failed** (§5.2) | 21.0 s |
| `test_audit2609_a10_io`, `a10_codev`, `audit_lens`, `w5_raytrace_bundles` | 106 passed, 4 skipped, **1 failed** (the same one) | 8.3 s |
| `test_niche_audit_w3_oracles.py` | 177 passed, **4 failed** — collateral, see below | 102 s |
| `prefix_check_verify.py` (fail-before) | **13 of 15 fail pre-fix**, 2 are the intended controls | 3 s |
| the 11 `repro/UI` scripts | §6 | — |

**Skip set unchanged.**  38 skips with and without my file, in either collection
order.  My module reuses `test_audit2609_a9_ui`'s stub install/park bootstrap,
so there is still exactly one owner of it.

**Pre-existing failures, unrelated to WP-A9 or to this pass** (4, all in
`tests/unit/test_niche_audit_w3_oracles.py`):
`test_w3_t3b_lg_merit_responds_to_a_curvature_change`,
`test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged`,
`test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`,
`test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged`.  Identical to the
four WP-A9 reported.  The assertion sites are in the test file itself
(`tests/unit/test_niche_audit_w3_oracles.py:2792`: `(12310011.487876404
−59649449.46912211j) != (15.44807582649+3.188059447022j)`, a σ-grid value off by
6e6 relative); the values come from `lumenairy.LGAberrationMerit` →
`lumenairy/propagators/asymptotic.py::fit_canonical_polynomials` /
`aberration_tensor`, and `asymptotic.py`, `asymptotic_canonical_fit.py`,
`asymptotic_aberration_tensor.py`, `asymptotic_maslov.py`,
`asymptotic_jax_twin.py` are all **modified in the working tree** by the
concurrently running asymptotic WP.  `lumenairy/ui/*` is not on that import
path, and my `raytrace/trace.py` change is provably inert for the fixture: the
test's `_singlet_t3b` is `make_singlet(...)`, which emits **no `elements` key at
all**, so the branch I touched is never entered.  **Collateral, not mine.**

The single `test_audit_w5_raytrace_bundles` failure is §5.2 — a test that pins
the U2 defect and must be restated in the same commit as §2.4 and §5.1.

---

## 8. Open items for the orchestrator

| # | severity | item |
|---|---|---|
| 8.1 | P3 | `MainWindow._ins_source_preset` installs an unknown kind string verbatim (`SourceDefinition('not_a_kind')` succeeds and every downstream `source_type` branch falls through to its default).  Only the six wired menu actions reach it today.  Suggested: validate against the combo's own list and raise with the §2 prefix. |
| 8.2 | P3 | `SurfaceDetailPanel._apply_source_params` writes back all 13 numeric fields from the form widgets, which `_load_source_ui` populates when the Source row is selected.  A source installed programmatically **while that row is already selected** (an Insert ▸ Source preset, a session restore) is therefore overwritten from stale widget text on the user's next keystroke.  Not reproducible headlessly — needs a real PySide6 to confirm whether the selection refresh covers it. |
| 8.3 | P3 | `SourceDefinition` puts no upper bound on `emitter_nx` / `emitter_ny`: `1e9` is accepted and `to_source` then loops 10¹⁸ times.  A `<= 4096`-style cap with the §2 prefix would make it a named error instead of a hang. |
| 8.4 | P3 | `lumenairy/ui/_mpl.py`'s PEP-562 hook raises whatever the import raises, so `hasattr(_mpl, 'FigureCanvasQTAgg')` propagates `ModuleNotFoundError: shiboken6` under the stub instead of returning `False`, and any `dir()`-driven introspection (`inspect.getmembers`, `help()`, a REPL completion) forces the Qt backend import.  Suggested: catch `ImportError` in `__getattr__` and re-raise as `AttributeError` with the original chained. |
| 8.5 | P3 | An inverted Start/End element range produces a successful run over zero surfaces (`range='empty element range'`, one saved plane) rather than an error payload.  The summary does say so. |
| 8.6 | P3 | Interrupting an `OptimizeWorker` returns `(False, 'Merit: … after 2 iterations [Nelder-Mead]')` rather than "Cancelled": `run_optimization` swallows the callback's `StopIteration` internally, so the worker's own `except StopIteration` never fires.  The cancellation itself works (2 of 200 iterations) and `success=False` keeps the result from being applied. |
| 8.7 | P2 | `tests/unit/test_v5_4_2_run_trace_empty_prescription.py` and `test_v5_4_6_io_ui_delegated.py` still `pytest.skip` on PySide6's absence — the TESTING_STANDARDS §4 shape WP-A9's own file was written to avoid.  Converting them to the stub harness is a separate job: it changes the 38-skip invariant the orchestrator is currently using as a regression signal, so it should be one deliberate commit. |
| 8.8 | P3 | `SystemModel.object_distance_m()` returns `inf` when a point source has no placed optic and the form field is `inf`, and `to_prescription` exports `object_distance: inf`.  Consumers gate on `> 0`, so an infinite conjugate reads as a finite one.  Suggested: treat non-finite as 0.0 (= at infinity). |
| 8.9 | P3 | `_build_trace_surfaces_internal` closes an element's air gap against `elements[ei + 1]` while `_build_trace_surfaces_world` uses the new `_next_optical_element(ei)`.  They agree for every list the GUI can build (Source/Detector are the only non-optical entries and only ever sit at the ends); an element with an empty `surfaces` list in the middle would make them disagree. |

---

## 9. Files touched by this pass

**Modified (4).**
`lumenairy/raytrace/trace.py` (the granted U2 block, lines 534–586),
`lumenairy/ui/waveoptics_dock.py` (`_filter_wave_optics_surfaces`,
`_prescription_from_surfaces`), `lumenairy/ui/optimizer_dock.py`
(`GlobalSearchWorker.run` / `_run_impl`), `lumenairy/ui/model.py`
(`set_display_distance`, new `_display_distance_slope`).

**New (1).** `tests/unit/test_audit2609_a9_verify_ui.py` (34 tests).

**Docs (2).** this file;
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A9_CHANGELOG.md`
(appended: one BLOCK 1 entry for the `trace.py` U2 finish, three BLOCK 2 entries
for the UI defects).

My `raytrace/trace.py` diff is a **single hunk** (49 +/15 −, lines 531–589) —
nothing else in that file is mine.  `lumenairy/raytrace/__init__.py` and
`lumenairy/raytrace/surface.py` picked up working-tree edits from a
concurrently-running agent partway through this pass (an `exit_vertex` re-export
block and a `_base_surface_sag_xy` addition); every raytrace number quoted above
was re-measured after they landed, and the raytrace files are green except the
one pin in §5.2 (155 passed / 1 failed at the end of the pass).

No git write commands were run.  Scratch scripts (`a_u1.py`, `a_u2_impact.py`,
`a_u2_shapes.py`, `a_u3.py`, `a_u4_u5.py`, `a_u5_gap.py`, `a_u6.py`, `a_u7.py`,
`prefix_check_verify.py`) live in the session scratchpad and are not committed.

---

# Follow-up (2026-09-12) — companions landed, open items closed

Requested by the orchestrator after the pass above.  Everything below is
implemented, measured and pinned; **0 failures remain in every file listed.**

## F1. The two companions to the `trace.py` U2 finish  [§5.1, §5.2 — DONE]

**`lumenairy/raytrace/jax_trace.py::_resolve_semi_diameters`** now carries the
same precedence and the same shape-aware index as `trace.py`, and its docstring
rule 3 is restated ("consulted ONLY when the per-surface key is absent or
invalid … which entry matches surface *i* depends on the prescription's shape").
Cross-backend parity re-measured on the four shapes that exercise every branch:

| prescription | NumPy | JAX | parity |
|---|---|---|---|
| per-surface key present, looser+tighter `elements` | `[0.002, 0.001]` | `[0.002, 0.001]` | ✔ |
| lens-only `surfaces`, mirror in `elements` | `[0.006, 0.006]` | `[0.006, 0.006]` | ✔ |
| chronological `surfaces`, no per-surface key | `[0.0127, 0.006, 0.006]` | `[0.0127, 0.006, 0.006]` | ✔ |
| designer folded export (the U2 case) | `[0.025, 0.006, 0.006]` | `[0.025, 0.006, 0.006]` | ✔ |

Before this change the JAX resolver returned `[0.001, 0.001]` and
`[0.006, 0.006, 0.006]` on rows 1 and 4 — the two backends vignetted the same
prescription differently.

**`tests/unit/test_audit_w5_raytrace_bundles.py`** —
`test_resolver_precedence_per_surface_then_elements_min` is restated as
`test_resolver_precedence_per_surface_wins_over_elements` with the requested
docstring and `== [2e-3, 1e-3]`.  I added two more parity pins in the same file
rather than leave the new index rule untested there:
`test_resolver_both_elements_shapes_agree_across_backends` (both layouts, both
backends) and `test_resolver_folded_designer_export_agrees_across_backends` (the
producer the finding is about).  The other three resolver tests are unchanged
and still pass.

**Runs.** `test_raytrace`, `test_audit_raytrace`, `test_audit2609_a1_raytrace`,
`test_audit2609_a1_exit_vertex`, `test_audit2609_a1_verify_oracles`,
`test_audit_w3_raytrace_parity`, `test_audit_w5_raytrace_bundles`,
`test_audit_w6_raytrace`, `test_niche_audit_w3_raytrace_sources`,
`test_v5_4_1_raytrace_mirror_backward_ray`,
`test_v5_4_2_run_trace_empty_prescription`, `test_v5_4_6_wave8_raytrace`,
`test_v5_4_retrace_ghost_path`, `test_niche_audit_a1_radial_metrics` and
`test_v5_4_7_walker_v20_cross_backend_parity` together:
**314 passed, 0 failed, 0 skipped, 50.0 s.**  `test_audit2609_a10_io`,
`a10_codev`, `audit_lens`, `v5_4_6_io_ui_delegated`: **95 passed, 4 skipped**
(the four are `test_audit_lens.py`'s own GUI skips, a file outside this pass).

## F2. The UI open items  [8.1, 8.3, 8.4, 8.5, 8.6, 8.8, 8.9 — DONE]

One pin each, in `tests/unit/test_audit2609_a9_verify_ui.py`, and each replayed
against the pre-fix code in process (`prefix_check_followup.py`):
**7 of 7 fail pre-fix, 0 vacuous.**

| item | change | measured before → after | pin |
|---|---|---|---|
| **8.1** | `main_window._ins_source_preset` validates `kind` against `SourceDefinition.TYPES` (the list the source-type combo is built from) and raises a §2-prefixed `ValueError` | `'not_a_kind'` installed verbatim → `ValueError: _ins_source_preset: unknown source kind 'not_a_kind'; expected one of [...]`, and the live source is untouched | `test_followup_81_source_preset_rejects_an_unlisted_kind` (also walks all 7 listed kinds) |
| **8.3** | `model._MAX_EMITTER_COUNT = 4096`, enforced in `_as_count` with a §2-prefixed message that says why; non-finite rejected too | `emitter_nx=1e9` accepted → named `ValueError`; 4096 still accepted, 4097 / 1e9 / 1e18 / `inf` / `nan` rejected | `test_followup_83_emitter_counts_are_capped` |
| **8.4** | `_mpl.__getattr__` catches `ImportError` and re-raises `AttributeError` with the original **chained** (`from exc`) | `hasattr(_mpl, 'FigureCanvasQTAgg')` raised `ModuleNotFoundError: No module named 'shiboken6'` → returns `False`; `inspect.getmembers(_mpl)` now works; `_mpl.Figure` still resolves and an unknown name is still a plain `AttributeError` | `test_followup_84_mpl_shim_raises_attributeerror_not_importerror` |
| **8.5** | an inverted / empty Start-End range emits an error payload naming the range and the available optical elements, instead of propagating zero surfaces | `start=3, end=1` → a "successful" 1-plane result → `{'error': 'Start/End element range selects no optical element (start=3, end=1; optical elements are [1, 2])'}`; a valid sub-range still runs and still reports `elements 2..2` | `test_followup_85_inverted_element_range_is_an_error_not_an_empty_run` |
| **8.6** | `OptimizeWorker.run` re-reads both cancellation flags **after** `run_optimization` returns (that method's own `except Exception` swallows the callback's `StopIteration`, so the worker's `except StopIteration` never fires) and emits `cancelled` + a "Cancelled by user" message | `(False, 'Merit: 89.821 um RMS spot after 2 iterations [Nelder-Mead]')` → `(False, 'Cancelled by user -- best so far: Merit: …')` with `cancelled` emitted; an uninterrupted run still reports its merit normally | `test_followup_86_interrupted_optimizer_reports_cancelled` |
| **8.8** | `object_distance_m()` returns 0.0 for any non-finite or non-positive distance, on both the element-geometry and form-field branches | form field `inf` → `object_distance_m() == inf` and `to_prescription()['object_distance'] == inf` (which every `> 0` gate accepts as a finite conjugate) → `0.0` on both; a finite field is still honoured, and an infinite FIRST-ELEMENT distance still falls through to the field | `test_followup_88_non_finite_object_distance_reads_as_infinity` |
| **8.9** | `_build_trace_surfaces_internal` closes its air gap against `_next_optical_element(ei)`, the helper the world builder already used | with a surface-less element between two optics the local list dropped the 40 mm gap the world list kept, so the two ABCDs disagreed → thicknesses, `find_paraxial_focus` and `system_abcd` identical from both lists; **bit-identical** (`[0.003, 0.04, 0.003, 0.0]`) on an ordinary list | `test_followup_89_local_and_world_lists_close_the_same_gaps` |

`tests/unit/test_audit2609_a9_verify_ui.py` is now **41 tests** (34 + 7).

**8.2 remains open and needs real PySide6.** `SurfaceDetailPanel._apply_source_params`
writes all 13 numeric source fields back from the form widgets, which
`_load_source_ui` repopulates when the Source row is selected.  Whether a source
installed programmatically *while that row is already selected* (an
Insert ▸ Source preset, a session restore) triggers that refresh before the
user's next keystroke is a Qt selection/signal question the stub cannot answer —
it has no widget, no selection model and no event loop.  Verifying it needs a
PySide6 box: select the Source row, fire `Insert ▸ Source ▸ Fiber mode`, then
edit one visible field and assert `fiber_mfd_um` is still 10.4 rather than the
stale widget text.

**New open item found while pinning 8.8** [P3]: `recompute_element_frames`
computes `origin += d * R[:, 2]`, so a non-finite `Element.distance_mm`
multiplies `inf` by the axis's zero components and leaves a **NaN origin** on
that element and every one after it (numpy emits
`RuntimeWarning: invalid value encountered in multiply`).  The Distance column
cannot produce a non-finite value, so this is reachable only from a scripted or
hand-edited session; the pin suppresses the warning explicitly rather than
hiding it.  A `np.isfinite` guard in the mutators would close it.

## F3. Converting the two skip-on-PySide6 files to the stub harness  [8.7 — DONE]

`tests/unit/test_v5_4_2_run_trace_empty_prescription.py` (2 skips) and
`tests/unit/test_v5_4_6_io_ui_delegated.py::test_richards_wolf_dock_compute_runs`
(1 skip) now run on the auditor's Qt stub.  Both reuse
`test_audit2609_a9_ui`'s install/park bootstrap rather than standing up a second
copy, so there is still exactly one owner of that discipline and the other ten
files' `try: import PySide6` guards still see the interpreter they had before.

**New skip invariant: 35** (was 38).  Measured three ways, all identical:

| set | before | after |
|---|---|---|
| the 12 sibling UI files alone | 47 passed, **38 skipped** | 50 passed, **35 skipped** |
| the 12 + both A9 files | 117 passed, **38 skipped** | 127 passed, **35 skipped** |
| …reversed collection order | 117 passed, **38 skipped** | 127 passed, **35 skipped** |

The three recovered pins are
`test_run_trace_empty_prescription_emits_trace_ready_none`,
`test_run_trace_empty_prescription_does_not_set_wait_cursor` and
`test_richards_wolf_dock_compute_runs` — all three pass.  The remaining 35 skips
are the other ten files' own `pytest.importorskip` / `skipif` guards
(`test_audit_w5_ui` 16, `test_audit_w6_ui` 5, `test_audit_s4_7_*` 5,
`test_audit_p1_gui_dead_import` 3, `test_audit_s4_3_*` 2, `test_audit_s4_2_*` 1,
`test_audit_s4_6_*` 1, `test_v5_14_5_*` 1 module-level); converting those is a
larger job — they construct real widgets, not model-level physics — and is not
part of this request.

## F4. Still deferred

**§5.3** (`user_library.load_lens` back-fill of `is_mirror` / `semi_diameter` /
`is_stop` onto folded prescriptions saved before the U1 fix) stays listed and
unstarted: `lumenairy/user_library.py` is WP-A15b's now.  §2.4 already closes
half of it — a pre-fix library entry's mirror resolves to its own aperture
rather than the following lens's — so what remains is the `is_mirror` flag
itself.

## F5. Complete file list for this pass (original + follow-up)

**Library (7).**
`lumenairy/raytrace/trace.py` — the U2 semi-diameter block (one hunk, lines
531–589);
`lumenairy/raytrace/jax_trace.py` — `_resolve_semi_diameters` + docstring rule 3;
`lumenairy/ui/model.py` — `set_display_distance` + new `_display_distance_slope`,
`_MAX_EMITTER_COUNT` + `_as_count`, `object_distance_m`,
`_build_trace_surfaces_internal`'s gap close;
`lumenairy/ui/waveoptics_dock.py` — `_filter_wave_optics_surfaces`,
`_prescription_from_surfaces`, the empty-range error payload;
`lumenairy/ui/optimizer_dock.py` — `GlobalSearchWorker.run`/`_run_impl`,
`OptimizeWorker.run`'s post-call cancel check;
`lumenairy/ui/main_window.py` — `_ins_source_preset` validation;
`lumenairy/ui/_mpl.py` — `__getattr__`'s ImportError translation.

**Tests (4).**
`tests/unit/test_audit2609_a9_verify_ui.py` (new, 41 tests);
`tests/unit/test_audit_w5_raytrace_bundles.py` (1 test restated, 2 added);
`tests/unit/test_v5_4_2_run_trace_empty_prescription.py` (stub harness);
`tests/unit/test_v5_4_6_io_ui_delegated.py` (stub harness).

**Docs (2).** this file;
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A9_CHANGELOG.md`.

I did **not** touch `lumenairy/raytrace/__init__.py` or
`lumenairy/raytrace/surface.py` (WP-A15b's), nor `lumenairy/user_library.py`.
No git write commands were run at any point.

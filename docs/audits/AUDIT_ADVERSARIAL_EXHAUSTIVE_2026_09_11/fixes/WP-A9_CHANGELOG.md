# WP-A9 changelog text — Designer UI (`lumenairy/ui/`), audit findings U1–U7

Two separate blocks below.  The first is for the **library** `CHANGELOG.md`
(effects a non-GUI user of `lumenairy` can observe); the second is for
**`GUI_CHANGELOG.md`** (the designer's own log).  The orchestrator should copy
each block into its own file verbatim.

---

## BLOCK 1 — for `CHANGELOG.md`

### Fixed -- designer UI: `SystemModel.to_prescription()` now exports the system the layout shows (audit U1, U2)

`lumenairy/ui/model.py` `to_prescription()` emitted only
`radius / conic / aspheric_coeffs / glass_before / glass_after / radius_y /
conic_y / aspheric_coeffs_y` on each legacy `surfaces` entry.  `is_mirror`,
`semi_diameter` and `is_stop` were all dropped, and the coord-break Surfaces
were `continue`d with their transfer thickness.  Consequences in the exported
dict — the single prescription handed to 18 call sites across 14 docks,
including the user's saved lens library:

* a mirror round-tripped as an air→air refracting surface (a no-op) while its
  Zemax-signed **negative** post-mirror gap survived, so the negative thickness
  became a literal backwards propagation;
* `surfaces_from_prescription` fell back to matching the `elements` list, whose
  refracting-surface filter skips mirrors while its index counts them, so every
  mirror shifted the aperture mapping by one and the trailing surface fell
  through to `aperture_diameter / 2`;
* a 45° fold lost its entire mirror-to-next-element air gap.

Measured on the audit's fixture (concave mirror R = −200 mm + N-BK7 singlet),
exported prescription vs the model's own internal surface list:

| quantity | before | after | oracle |
|---|---|---|---|
| EFL | 158.862 mm | **178.774 mm** | 178.774 mm (`build_trace_surfaces()`) |
| BFL | 156.201 mm | **127.802 mm** | 127.802 mm |
| trailing-surface semi-diameter | 12.7 mm | **6.0 mm** | 6.0 mm (the surface's own) |
| 45° fold total axial gap | 3 mm | **43 mm** | 43 mm |

`to_prescription()` additionally emits `object_distance` (the geometric
source-to-first-optic gap, 0.0 for a collimated source), `image_distance` (last
surface to Detector) and `stop_index`.  `analysis.eval_image_plane_wfe` and
`analysis.plotting`'s finite-conjugate ray launch consume `object_distance`, so
a point-source design is no longer analysed as if it were collimated.
Byte-equivalence of the unfolded singlet export to `make_singlet` is unchanged
(EFL 99.288517 mm / BFL 97.293283 mm from both, to 15 digits).

New: `SurfaceRow.is_stop`, carried through `load_prescription` (from either a
per-surface `is_stop` or a `stop_index`), the two trace-surface builders, the
session JSON and `to_prescription`.  The .zmx `STOP` keyword previously could
not survive an import → edit → export round trip at all.

Tests: `tests/unit/test_audit2609_a9_ui.py::test_u1_*`, `::test_u2_*`.

### Fixed -- designer UI: a wave-optics run no longer disables pyFFTW / SciPy FFT for the whole process (audit U4)

`ui/waveoptics_dock.py`'s worker set `fft_infra.USE_PYFFTW = USE_SCIPY_FFT =
False` unconditionally at the top of every run — from the worker thread, on
module globals shared with every other dock, the REPL dock and anything the user
scripts — and never restored them.  The backend combo's index 0 was `NumPy FFT`,
so one **default** Run silently downgraded every subsequent FFT in the session
from multi-threaded SciPy/pyFFTW to single-threaded numpy, with no UI
indication.  `set_max_ram(mem_limit)` had the same shape.

Both are now scoped to the run by a `try/finally` context manager
(`_process_overrides`), restored on every exit path including an exception, and
the combo gained a new index-0 entry **"Library default"** that touches no global
at all.  `tests/conftest.py`'s cross-module flag leak-guard comment cites this
exact site as the reason `fft_infra` is in `_LEAK_GUARD_MODULES`; the leak it
describes no longer exists (the guard itself is still worth keeping).

Test: `tests/unit/test_audit2609_a9_ui.py::test_u4_*`.

### Fixed -- designer UI: `lumenairy.ui` no longer imports matplotlib at module scope (performance)

14 dock modules imported `matplotlib.figure` and
`matplotlib.backends.backend_qtagg` at module scope, so importing ANY of them —
including from a headless test harness — pulled the whole matplotlib stack
(`python -X importtime -c "import matplotlib.figure"`: ~17.5 s cumulative cold on
this workstation, sub-second warm).  They now go through a new
`lumenairy/ui/_mpl.py` PEP-562 shim that imports on first figure construction,
so a dock the user never opens costs nothing.  `_mpl.style_axes(ax)` also
replaces the `set_facecolor` / `tick_params` / per-spine `set_color` triple that
was copy-pasted into ~20 docks.

Measured: importing 15 dock modules leaves `matplotlib.figure` out of
`sys.modules` (pinned by
`tests/unit/test_audit2609_a9_ui.py::test_u7_matplotlib_is_not_imported_by_the_dock_modules`).
`import lumenairy` still does not pull `lumenairy.ui` (also pinned).

### Changed -- designer UI: the point-source object distance comes from the element geometry

`SourceDefinition.object_distance_mm` and the first element's `distance_mm` were
two independent knobs for one quantity: rays launch from the Source element at
world z = 0 and the first surface sits at its own `distance_mm`, so the pupil
fill was correct only when the user happened to keep the two equal.  The ray
launch and the exported `object_distance` now both use the geometric gap
(`SystemModel.object_distance_m()`); the form field is honoured only when no
optic has been placed yet, and its tooltip says so.

**Migration.** A design whose source form said e.g. 1000 mm while the first lens
sat at 200 mm previously under-filled the entrance pupil by 5×; it now fills it
correctly.  To reproduce the old numbers, move the first element to the distance
the form field named.

### Added -- `lumenairy/ui/_worker.py`

`AnalysisWorker` (a `QThread` base with a non-shadowing `finished_result`
signal, a guaranteed single emission on every path, and a `check_interrupt()`
helper), `ThreadCancellableProgress` (a `CancellableProgress` that is also
tripped by Qt's `requestInterruption()`), and `interrupt_check(worker)`.

---

## BLOCK 2 — for `GUI_CHANGELOG.md`

### Fixed -- analysis docks were analysing a different system than the layout drew (U1, U2, P0)

Every dock that calls `SystemModel.to_prescription()` — Analysis, Caustic,
Coherence, Ghost, Jones-pupil, LG-aberration, **Library (save)**, Multi-config,
Optimizer, PSF/MTF, Snapshots, Wave-optics, and six entry points in the main
window — received a prescription with `is_mirror`, `semi_diameter` and `is_stop`
stripped and folded-system air gaps deleted.  On the audit's fold fixture the
docks computed EFL 158.9 mm where the layout, the spot diagram and the status
bar showed 178.8 mm; a 50 mm fold mirror was vignetted to 12 mm; a 45° fold's
40 mm gap vanished.  All three keys are now emitted per surface and the
coord-break gaps are folded into the surrounding air gap.

Remaining library-side item (reported to the orchestrator, not fixable from
`ui/`): `raytrace/trace.py:546-552` takes `min(per-surface semi_diameter,
elements[i] semi_diameter)` with a refracting-only `elements` filter indexed by
the all-surfaces counter, so a mirror is still clipped to the following lens's
semi-diameter.  Every refracting surface is now exact.

### Fixed -- point-source ray bundles were degenerate (U3)

With `source_type = 'point_source'` selected (the Insert ▸ Source preset
defaults it to 1000 mm) the azimuth loop variable was never used: all rays were
launched with `L = M = rho`, so `num_rings × rays_per_ring` rays collapsed onto
`num_rings` distinct directions on the x = y diagonal and the marginal ray was
`sqrt(2)` = 41 % too steep.  The spot diagram, `spot_rms`, `spot_geo_radius`,
the ray fan and the 2-D ray overlay were all computed from that.  Measured at
`num_rings = 3, rays_per_ring = 8`: **4 unique directions → 25**, marginal
|rho| 0.089803 → **0.063500** (exactly `semi_ap / obj_dist`).

### Fixed -- one wave-optics run no longer downgrades the FFT backend for the rest of the session (U4)

See the library block.  The Backend combo gained a **"Library default"** entry at
index 0; every explicit choice is applied for the duration of the run and
restored afterwards, as is the Memory-limit cap.

### Fixed -- the wave-optics summary now reports which lens model actually ran (U5)

The lens-model router caught EVERY exception from `apply_real_lens[_traced|
_maslov]` and fell through to the crude per-surface ASM loop in silence, while
the summary kept printing the method the user picked.  A folded design makes
those functions refuse **by design**, so every fold silently produced a
thin-screen PSF labelled "Real lens (traced)".

Now: with **Unfold mirrors** ticked the router hands the core function the
unfolded-equivalent prescription the per-surface loop would have walked
(mirrors removed, their gaps carried, Zemax-signed thicknesses flipped
positive), so the requested model really runs and the summary says
`Lens model: real_lens_traced (unfolded equivalent)`.  With the box unticked, or
on any other failure, the summary says
`Lens model: asm (fallback)  (requested real_lens_traced)` plus a one-line
reason.  `results['lens_model_used'] / ['lens_model_requested'] /
['lens_model_fallback_reason']` carry the same information to other docks.

Related: `_filter_wave_optics_surfaces` used to drop a mirror Surface *and* its
thickness, silently shortening the unfolded path by the whole
mirror-to-next-element gap (30 mm on the audit fixture).  The gap is now carried
onto the previous kept surface; total unsigned axial path is conserved.

### Fixed -- U6 crash / wrong-wavelength list

* **`_on_finished` sliced the focal field with the INPUT grid N** — an
  `IndexError` on the three `*-mft` methods and any detector run (where
  `I_focus` is `mft_N_out²` or re-binned to the pixel pitch).  It now slices the
  OUTPUT array and uses the output `dx`; the worker publishes `N_out` / `dx_out`
  on every path and the summary shows `Grid: 256x256 in -> 64x64 out`.  The PSF
  imshow is also strided down to ≤ 512 px (an N = 8192 run was drawing a 2048²
  image).
* **Insert ▸ Source ▸ … (six menu actions) raised `AttributeError`** —
  `SystemModel.source` is a read-only property and the handler assigned to it,
  with no `try` around it.  All six presets now go through `set_source()`, which
  checkpoints for undo, syncs the wavelength, invalidates and emits.
* **Every source-parameter edit reset the wavelength to 1310 nm and dropped the
  polarization** — the form has 13 line edits, neither of which is λ or the
  Jones state, so `SourceDefinition.__init__`'s defaults won each time.  Both
  are now carried forward, and `SystemModel.set_wavelength()` /
  `set_source()` / `load_prescription()` sync `source.wavelength_nm` to the
  model (which is authoritative).  Measured: max |phase difference| between the
  launched point-source field and the correct-wavelength one **0.216 rad → 3.5e-17
  rad** at 632.8 nm.
* **`emitter_array` was unusable** — the element table cast every field to
  `float`, including the two integer counts, so `to_source`'s `range()` raised
  `TypeError`; the wave worker's broad handler swallowed that into an EPD-clipped
  **plane wave** labelled "Emitter array 12×12", and the 2-D/3-D scene rebuild
  raised on the same value.  The counts are cast to `int` in the form,
  `SourceDefinition` validates them with the `§2` error prefix, and both layouts
  guard with `int()`.
* **`_build_trace_surfaces_world()` dropped every inter-element air gap** — the
  world list is what `run_trace`'s paraxial-focus fallback and six docks
  (footprint, ray fan, spot field, tolerance, distortion, wavefront map) take
  ABCDs on.  Measured on two singlets 40 mm apart: `find_paraxial_focus`
  **58.344 mm → 40.112 mm** (the local list's value, exactly), `system_abcd` EFL
  **61.577 mm → 72.971 mm**.  With the Detector at 0 the image plane had been
  landing 18 mm past focus and every spot / RMS / Airy number was taken there.
* **PSF/MTF "Pupil from ray trace" was dead and, once revived, wrong** — it read
  `rays.opl` (`RayBundle` has only `opd`), so it always died as "Pupil load
  failed: AttributeError"; and it binned `image_rays.x/y` — positions at the
  IMAGE plane, max |r| = 0.25 mm for a 25.4 mm pupil — into an EPD-wide grid, 31
  of 65536 cells.  It now reads `opd`, takes the ray history at the last real
  optical surface, and references it to that surface's vertex plane with the
  shared `exit_vertex_transfer` operator.  Measured: pupil radius **0.25 mm →
  12.7 mm** (= EPD/2, to 2 %).
* **The optimizer worker mutated the live model on every scipy probe** —
  `merit_function` → `set_variable_values` → `_invalidate` →
  `recompute_element_frames` rewrote every element's `origin` / `R` and nulled
  `_flat_surfaces_cache`, hundreds of times a second, from the worker thread,
  while the GUI painted from exactly those attributes.  The documented
  `apply_result=False` fix only governed the final write-back.  Both
  `OptimizeWorker` and `GlobalSearchWorker` now optimize a detached deep copy and
  hand only `result_x` back for the GUI thread to apply.
* **14 of 16 QThread workers ignored `requestInterruption()`** — so
  `MainWindow.closeEvent`'s `requestInterruption()` + `wait(2000)` timed out and
  Qt aborted the process during shutdown, truncating any in-flight HDF5/Zarr
  write.  Every worker now has a cancellation path: the four that own a
  `CancellableProgress` use the new `ThreadCancellableProgress`, which reads
  Qt's flag as well; the rest poll at entry and (coronagraph, AO) at each stage
  boundary.  Pinned by a structural sweep over all 18 `QThread` subclasses.

### Fixed / Changed -- U7 (P2 / P3)

* `OptimizeWorker`, `GlobalSearchWorker` and `ToleranceWorker` no longer shadow
  `QThread.finished` (renamed to `finished_result`, matching the other 13); the
  canonical `worker.finished.connect(worker.deleteLater)` idiom works again.
  Pinned by a sweep over every worker class.
* **"Start at / End at"** in the wave-optics dock now actually restricts the
  propagation.  Both locals were read from the config and never referenced
  again, so the user narrowed the run and silently got the full system.  The
  range maps through a new `SystemModel.element_surface_spans()`; a restricted
  run re-derives its own paraxial focus, reports `Range: elements 1..2 (surfaces
  0..3)` in the summary, and declines the whole-prescription lens router (which
  cannot honour a sub-range) with that as the stated reason.
* `power_in` is captured from the launched field immediately after
  construction.  It was read back from `planes[0]`, which is only the source
  plane when "Source" is ticked in the save list — untick it and "Throughput"
  silently became ~100 %.
* `build_run_trace_world_surfaces(image_distance=…)` honours an explicit
  argument over the Detector element.  The Detector branch was tested first, so
  `spot_field_dock` (which passes the paraxial BFL) and `distortion_dock` drew
  focal-plane overlays — Airy radius, distortion grid — on a diagram rendered at
  the detector plane: 100 mm vs a 97.3 mm BFL on a default new system.
  `image_distance=None` keeps the Detector preference.
* `get_variable_values()` / `set_variable_values()` walk the same filtered list.
  The getter skipped stale entries (returning a shorter vector) while the setter
  indexed `values[i]` over the unfiltered list, so deleting an element that owned
  an optimization variable made the next optimize assign values to the wrong
  parameter — or raise `IndexError`, swallowed into a generic failure message.
  A wrongly sized vector is now a named `ValueError`; `delete_element` re-bases
  the surviving indices and `move_element` follows the swap; the optimizer's
  variable grid, the multi-config dock and the global search all read the live
  list.
* **File ▸ New** calls a new `SystemModel.reset_design()` instead of
  `self.model.__init__()` on the live QObject.  Re-running `__init__` reset
  `prefs` (2-D/3-D backgrounds, ray colour, ray-use-wavelength, accent, theme),
  `lens_options`, `auto_retrace_mode`, `unit_preference` and `snapshots` — none
  of which "New system" should touch — and re-invoked `QObject.__init__` on an
  already-constructed C++ object.
* The forecast panel's ASM calibration runs on a worker thread.
  `_local_asm_baseline_ms()` ran three 512² complex FFT pairs inside a widget
  signal handler wired to 14 controls, freezing the dock on the user's first
  keystroke in it; the forecast now takes the 12 ms fallback until the background
  measurement lands, and "Recalibrate" is likewise asynchronous.
* `_apply_real_lens_asm_equiv` reproduces its own docstring's calibration
  points.  It is a cost-model coefficient for the Time estimate only (not a
  physics quantity and not an ASM step count): the shipped formula returned 1.4
  and 2.6 where the docstring stated 1.1 and 2.2, so `real_lens` /
  `real_lens_traced` forecasts read ~20 % pessimistic.
* `set_display_distance` routes through `_prev_element_back_vertex_world()`
  rather than re-deriving it inline, and `element_z_positions_mm()` reads the
  cached element frames instead of a naive cumulative sum of `distance_mm`.
  The two disagreed by each element's internal thickness, so in
  absolute-coordinates mode typing the DISPLAYED value back into the Distance
  column moved the element: measured 40.0 mm → **37.0 mm** per round trip on a
  3 mm-thick singlet.  The column now agrees with `Element.origin`, which is
  what both layout views and the absolute-position editor use.
* The `min_thickness` merit no longer carries a constant offset.  It iterated
  every element's trailing surface, whose thickness is 0 by the model's own
  convention (the air gap lives on `Element.distance_mm`), adding a fixed
  `(1 − 0)² = 1` per element: measured **1.0 → 0.0** on a fully compliant
  two-element design.
* The optimization completion message no longer labels every merit in µm.
  `Merit: {fun*1e6:.3f} um` is right for `rms_spot` and meaningless for
  `efl_target` / `bfl_target` / `seidel_spherical` / `min_thickness` /
  `max_fnumber`, which are dimensionless squared errors.
* The dead `_suppress_history` flag (set `False` in `__init__`, read in
  `_checkpoint`, never set `True` anywhere) is replaced by a working
  `SystemModel.bulk_edit()` context manager that groups a composite edit into
  one undo step.
* `layout_2d`'s emitter-array glyph no longer multiplies its column offset by
  `0.0` and redraws `nx` identical overlapping dots; the side view draws the y
  rows only, which is what the projection actually shows.
* `WaveOpticsWorker`'s model snapshot deep-copies the `Surface` objects, not just
  the list.  The class contract says "no reference into the model survives into
  `run()`"; `_filter_wave_optics_surfaces` only copied the entries whose sign it
  flipped, so the rest aliased the model's cached Surfaces.
* Session save/restore keeps the source polarization, the top-hat diameter and
  both fiber-mode fields.  `enc_source` listed 12 of the 16 constructor kwargs,
  so a save/restore silently reverted the other four to defaults.
* `analysis.py`'s "Image-plane WFE" panel is reachable again: it was gated on
  `presc.get('object_distance', 0) > 0` and `to_prescription()` never emitted
  that key, so the whole block was dead code.
* `diagnostics.report` uses `os.path.basename` instead of splitting on a
  hard-coded backslash (which printed the full path on POSIX).
* `run_lumenairy_designer.py`'s metres→nanometres guard accepts an int
  wavelength, not only a float.

### Added -- `ui/_worker.py` and `ui/_mpl.py`

`AnalysisWorker` / `ThreadCancellableProgress` / `interrupt_check` give the 16
hand-written worker copies one cancellation convention and one finished-signal
name.  `_mpl` makes matplotlib lazy and owns the shared dark-theme
`style_axes(ax)` helper.

---

## Tests

New: `tests/unit/test_audit2609_a9_ui.py` (36 tests).  It runs on the auditor's
Qt stub (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub`)
extended with permissive `QtWidgets` / `QtGui` shims — PySide6 is not installed
on this machine and the file does **not** skip on its absence.  The stub is
parked out of `sys.modules` between tests so the sibling UI test files, which do
guard themselves with `try: import PySide6`, keep making the same skip decision
they made before this file existed.

---

# VERIFY-A9 additions (independent verification pass, 2026-09-12)

Four defects found and fixed during the re-verification of WP-A9.  Blocks are
marked with their destination file, same convention as above.

## BLOCK 1 addition — for `CHANGELOG.md`

### Fixed -- raytrace: a per-surface `semi_diameter` is no longer clipped by the `elements` matcher (audit U2, completes WP-A9)

`raytrace/trace.py`'s `surfaces_from_prescription` read the per-surface
`'semi_diameter'` key and then unconditionally `min()`-ed it against
`prescription['elements']`, matched by index **within the refracting-surface
entries**.  A chronological `surfaces` list — the shape the designer UI exports —
includes its mirrors, so every mirror shifted that mapping by one and a fold was
clamped to the FOLLOWING lens's aperture.  Measured on the audit's fold fixture
(mirror semi-diameter 25 mm, lens 6 mm): the mirror resolved to **6.0 mm →
25.0 mm**; on a two-mirror design, **[8, 8, 8, 8] mm → [25, 20, 8, 8] mm**.

A present per-surface key now wins outright; the `elements` list is consulted
only as a fallback for producers that carry their apertures there (the .zmx /
CodeV loaders), and that fallback picks its index from the shape of the
prescription: positionally when `surfaces` itself carries mirrors and the two
lists are the same length, by the refracting-surface filter otherwise.  A
`semi_diameter` of `None` in an `elements` entry is now skipped rather than
raising `TypeError` on the `> 0` comparison.

Scope, measured: of the 62 prescriptions the repository can produce (six
builders, 15 `.zmx` fixtures, seven `.seq` fixtures, three designer exports and
`normalize_prescription` of each), **58 are bit-identical and the four that move
are the designer's folded exports** — the case the finding is about.

Tests: `tests/unit/test_audit2609_a9_verify_ui.py::test_u2_*` (including a
bit-identity guard that replays the pre-fix rule over every builder and loader).

**Companion changes this requires** (outside VERIFY-A9's ownership, listed in
`fixes/VERIFY_WP-A9.md` §5): `raytrace/jax_trace.py::_resolve_semi_diameters`
is the documented twin of this block and must take the same change or the two
backends disagree (measured: NumPy `[2e-3, 1e-3]` vs JAX `[1e-3, 1e-3]`), and
`tests/unit/test_audit_w5_raytrace_bundles.py::
test_resolver_precedence_per_surface_then_elements_min` pins the old
`min()`-against-a-present-key semantics and has to be restated.

## BLOCK 2 addition — for `GUI_CHANGELOG.md`

### Fixed -- unfolding a mirror put its air gap after the wrong surface (U5 follow-up)

`waveoptics_dock._filter_wave_optics_surfaces` carries a dropped mirror's axial
gap onto a neighbouring surface so the unfolded path keeps its length.  A
`Surface.thickness` is the gap **after** that surface, walked in the medium
after it, so the gap of a dropped surface belongs on the **preceding** kept
surface; it was being added to the following one.  On a
singlet-mirror-singlet design that moved 40 mm of fold air into the second
lens's 3 mm N-BK7 leg: thicknesses `[3, 25, 43, 0] mm` where the unfolded
equivalent is `[3, 65, 3, 0] mm`, and EFL/BFL **68.236 / 21.085 mm → 82.515 /
24.842 mm** (the hand-built unfolded list's values to the last bit, 17.3 % /
15.1 % apart).  A carry with no preceding kept surface is dropped (the field is
constructed AT the first surface, so a leading gap is not propagated — the same
rule the source-to-first-surface gap already follows) and a carry left over at
the end is flushed onto the last kept surface.

Related: `_prescription_from_surfaces`, which builds the unfolded-equivalent
prescription handed to the lens-model router, dropped a folded-out coord
break's transfer thickness with the Surface — the U1 defect one level down.  The
gap is now carried into the preceding gap, exactly as `to_prescription` does.

### Fixed -- the Optimizer dock's Global search never completed a single restart

`GlobalSearchWorker.run()` unpacked `opt_variables` — `(elem_idx, surf_idx,
field)` triples — into two names, so the first restart raised `ValueError: too
many values to unpack (expected 2, got 3)` **before any `finished_result`
emission**, leaving the dock's buttons disabled and its log at "Running...".
The conic test in the same loop keyed off `col_idx == 7`, a column index from a
table model this code no longer uses.  The loop now unpacks the triple and tests
the field name, and `run()` is wrapped so exactly one `finished_result` is
emitted on every path, including a merit function that always raises.

### Fixed -- the absolute-coordinates Distance column moved folded elements

The column shows `Element.origin[2]`, a world Z, while the value written back
is a distance along the optical axis.  The two coincide only while the axis is
parallel to world Z, so on a fold, typing the displayed value straight back
moved the element: measured 40 mm → **23.094 mm** on a 30° fold and 40 mm →
**0 mm** on a 45° one.  The write now divides by the z-component of the axis the
distance is measured along (which is exactly 1.0 on an unfolded system, so those
are bit-unchanged) and declines — without taking an undo checkpoint — when that
component is zero, i.e. when the leg runs perpendicular to world Z and the
column cannot describe the distance at all.

### Tests

New: `tests/unit/test_audit2609_a9_verify_ui.py` (34 tests) on the same Qt stub,
parked the same way; the sibling UI files' skip set is unchanged at 38 with and
without it, in either collection order.  13 of its 15 fix-pinning assertions
were replayed against the pre-fix code in process and fail there; the other two
are the deliberate unfolded controls.

---

# VERIFY-A9 follow-up (2026-09-12)

## BLOCK 1 addition — for `CHANGELOG.md`

### Fixed -- raytrace: the JAX aperture resolver follows the NumPy backend again (audit U2)

`raytrace/jax_trace.py::_resolve_semi_diameters` documents itself as mirroring
`trace.surfaces_from_prescription` key for key; the U2 precedence fix moved only
the NumPy side, so the two backends resolved the same prescription differently
(measured: NumPy `[2e-3, 1e-3]` vs JAX `[1e-3, 1e-3]` for a present per-surface
key, and `[0.025, 0.006, 0.006]` vs `[0.006, 0.006, 0.006]` for a folded designer
export).  The resolver now takes the same rule — a present per-surface
`semi_diameter` wins outright, and the `elements` fallback indexes positionally
for a chronological `surfaces` list and by the refracting-surface filter for the
lens-only `.zmx` / CodeV layout — and its docstring's numbered rule 3 says so.

Parity re-measured on the four prescription shapes that exercise every branch:
identical from both backends in all four.

Tests: `tests/unit/test_audit_w5_raytrace_bundles.py::
test_resolver_precedence_per_surface_wins_over_elements` (restated from
`..._then_elements_min`, which pinned the pre-U2 semantics),
`::test_resolver_both_elements_shapes_agree_across_backends` and
`::test_resolver_folded_designer_export_agrees_across_backends` (both new).

### Changed -- designer UI: `SourceDefinition` caps the emitter-array side at 4096

`to_source` builds an emitter array with a Python loop over
`emitter_nx * emitter_ny`, so an unbounded count was a hang rather than a slow
run: `emitter_nx = 1e9` was accepted and meant 1e18 iterations.  Counts above
4096 per side (and non-finite ones) are now a `ValueError` with the CONVENTIONS
§2 prefix that says why.  4096 x 4096 is 16.8 M emitters, far beyond anything
the designer can usefully model.

### Fixed -- designer UI: an infinite object distance is exported as "at infinity"

`SystemModel.object_distance_m()` passed a non-finite conjugate straight through,
and `to_prescription()` exported `object_distance: inf`.  Every consumer gates on
`object_distance > 0`, which `inf` satisfies, so an infinite conjugate was
solved as a finite one (`analysis.eval_image_plane_wfe`'s Gauss solve among
them).  Non-finite and non-positive both now report `0.0`, the prescription
convention for an object at infinity.

## BLOCK 2 addition — for `GUI_CHANGELOG.md`

### Fixed -- U7 follow-up items

* **Insert ▸ Source rejects an unknown preset.**  `_ins_source_preset` installed
  whatever string it was handed; an unlisted `source_type` matches no branch in
  `to_source` / `describe` / the layout glyphs and silently degrades to a plane
  wave.  The kind is now validated against `SourceDefinition.TYPES` — the same
  list the source-type combo is built from — with a §2-prefixed `ValueError`.
* **An inverted "Start at / End at" range is an error, not an empty run.**  It
  used to select zero surfaces and return a successful result: the source field
  pushed to the focus, i.e. a plausible-looking PSF of nothing.  The worker now
  emits an error payload naming the range and the available optical elements.
* **An interrupted optimize says "Cancelled".**  `run_optimization` wraps its
  scipy call in `except Exception`, and `StopIteration` is an `Exception`, so
  the sentinel the progress callback raises never reached `OptimizeWorker`'s
  `except StopIteration`: stopping a run mid-flight reported
  `Merit: 89.821 um RMS spot after 2 iterations` and read as a completed
  optimize.  Both cancellation flags are re-read after the call returns, and the
  worker emits `cancelled` plus `Cancelled by user -- best so far: ...`.
* **The local and world trace-surface lists close the same air gaps.**
  `_build_trace_surfaces_internal` tested only the immediately-next element
  while the world builder used `_next_optical_element`, so a surface-less
  element between two optics made the local list drop a gap the world list kept
  and the two ABCDs disagreed (40 mm on the fixture).  Both use the helper now;
  bit-identical on every list the GUI can build.
* `lumenairy/ui/_mpl.py`'s lazy-import hook re-raises an `ImportError` as an
  `AttributeError` with the original chained.  `__dir__` advertises the lazy
  names, so `hasattr`, `inspect.getmembers`, `help()` and REPL completion all
  reach it, and the raw `ModuleNotFoundError: shiboken6` escaped `hasattr` on a
  box without Qt bindings instead of making it return `False`.

### Tests

`tests/unit/test_audit2609_a9_verify_ui.py` grows to 41 tests (7 follow-up pins,
all demonstrated to fail against the pre-fix code in process).

`tests/unit/test_v5_4_2_run_trace_empty_prescription.py` and
`tests/unit/test_v5_4_6_io_ui_delegated.py::test_richards_wolf_dock_compute_runs`
no longer `pytest.skip` when PySide6 is absent — the shape
`docs/TESTING_STANDARDS.md` §4 forbids, which had removed three pins on exactly
the machines with no Qt.  They run on the auditor's Qt stub through the same
bootstrap the WP-A9 test file owns.  **The UI skip invariant is now 35, down
from 38**, identical with and without the A9 test files and in either collection
order.

## BLOCK 2 addition — for `GUI_CHANGELOG.md` (VERIFY-A9 follow-up, P3)

### Fixed -- a non-finite element spacing put NaN world origins on every element after it

`SystemModel.recompute_element_frames` advances the running origin by
`distance_mm * R[:, 2]` once per element.  An untilted axis is `(0, 0, 1)`, so
an infinite spacing multiplies `inf` by two zero components and the element's
world origin becomes `[nan, nan, inf]` — and, the walk being cumulative, so does
every element after it.  The 2-D/3-D layouts, both trace-surface builders and
every ABCD taken on them then read NaN, with nothing anywhere to say which
element caused it.  Measured on three singlets, `set_display_distance(2, inf)`:
elements 2, 3 and 4 all at `[nan, nan, inf]` and `find_paraxial_focus(world)`
= `inf`.  `nan` was worse still: `max(0, nan)` is `0` in Python, so a NaN entry
silently moved the element onto the previous one's back vertex.

The three mutators an operator or the optimizer can feed — the Distance column
in either coordinate mode (`set_display_distance`), the absolute-coordinates
editor's Z / X / Y columns (`set_element_absolute_field`) and the optimizer
write-back (`set_variable_values`) — now refuse a non-finite or non-numeric
spacing with a CONVENTIONS §2-prefixed `ValueError` naming the function, the
element and the value.  The two UI mutators validate before taking their undo
checkpoint, so a refused entry leaves no undo step.  Every finite write is
unchanged to the last bit, the `max(0, ...)` clamp on a negative entry included.

Test: `tests/unit/test_audit2609_a9_verify_ui.py::
test_followup_nonfinite_distance_is_refused_by_the_mutators` (inf / -inf / nan)
and `::test_followup_finite_distance_writes_are_bit_identical`.

## BLOCK 2 addition — for `GUI_CHANGELOG.md` (VERIFY-A9 follow-up, P3 continued)

### Fixed -- a non-finite tilt put a NaN rotation matrix on every element after it

The rotation half of the placement defect above.
`SystemModel.recompute_element_frames` accumulates both halves of each element's
world frame in one walk — `origin += distance_mm * R[:, 2]` and
`R = R @ Rx(tilt_x) @ Ry(tilt_y)` — so a non-finite tilt reaches `np.cos` /
`np.sin`, makes every entry of `R` NaN, and the NEXT element's `d * R[:, 2]`
carries the NaN into its origin as well.  Measured on three singlets, a NaN
`tilt_x` on element 2: `R[2, 2]` NaN on elements 2, 3 and 4, and
`origin = [nan, nan, nan]` on 3 and 4.

`SystemModel._as_distance_mm` is now a wrapper over a general
`_as_finite_element_field`, and the four operator-reachable placement writers
screen their input before taking any undo checkpoint: `set_element_field`
(cols 4-7), `set_element_absolute_field` (cols 3-7), and
`element_table.SurfaceFlatModel.setData` for both the tilt/decenter columns
(8-11) and the air-gap row's distance cell (col 4), the last of which bypassed
`set_display_distance` entirely and so still had the `max(0, nan) == 0` hole.
The refusal carries the CONVENTIONS §2 prefix and names the function, the
element, the field, its units and the value; in the two table writers it
surfaces as the cell reverting, exactly as a non-numeric entry already did.

Every finite write is unchanged: an unchanged value is still a no-change with no
checkpoint, an empty cell is still 0.0, a non-numeric entry still reverts, and
`set_element_absolute_field(elem, Rx, 30.0)` still gives an axis of exactly
`(0, -sin 30 deg, cos 30 deg)` to 1e-15.

Tests: `tests/unit/test_audit2609_a9_verify_ui.py::
test_followup_nonfinite_tilt_is_refused_by_the_mutators` (inf / -inf / nan,
driven through the real `SurfaceFlatModel.setData`) and
`::test_followup_finite_tilt_writes_are_bit_identical`.

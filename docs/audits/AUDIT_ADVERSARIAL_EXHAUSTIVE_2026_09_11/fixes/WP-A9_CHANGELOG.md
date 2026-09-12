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

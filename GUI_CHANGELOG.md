# Changelog — LumenAiry Designer (GUI)

All notable changes to the GUI application are documented here.
For core library changes, see `CHANGELOG.md`.

**Versioning note:** starting 2026-04-17 the GUI distribution tracks
the same version number as the core `lumenairy` library
(previously the GUI had its own 3.2.x track alongside the library's
3.1.x, which caused confusion -- the About dialog reads
`__version__` from the library, so users saw two different numbers
for the same release).  Historical GUI-only releases (e.g. 3.2.0,
2026-04-16) retain their original numbers below for traceability.

**Naming note:** the application was renamed from "Optical
Designer" to "**LumenAiry Designer**" in 3.5.9.  Earlier
historical entries below retain the old name for traceability.

## [3.7.8] — 2026-05-11

Closes the remaining 3.7.6 / 3.7.7 fold-correctness gaps and
ships three discoverability / workflow polishes.  All four
ray-trace docks (Footprint, Distortion, Spot vs Field, Ray
Fan) and the Tolerance Monte Carlo now route through the
world-frame trace.

### LumenAiry Designer is still in BETA

Layouts, dock arrangements, default workspaces, and even menu
names continue to change between minor releases.  Pin a
specific version if you depend on a particular GUI behaviour.

### Ray Fan dock -- tangential / sagittal / OPD plots are now
fold-accurate

3.7.7 migrated the Ray Fan dock's field-curvature plot to the
world-frame trace but left the main tangential / sagittal
ray-fan and OPD-fan plots on the legacy `ray_fan_data` /
`opd_fan_data` helpers (which used the local-frame
`trace()`).  3.7.8 plumbs the dock through the new
`ray_fan_data_world` / `opd_fan_data_world` core helpers, so
folded designs see the same `~1/cos(θ)`-corrected residuals
the layouts have been showing since 3.7.6.

### Tolerance dock -- world-frame Monte Carlo

3.7.7 documented the Tolerance dock as "intentionally not
migrated" because the Monte Carlo trial loop perturbs
`Surface.thickness`, which the world trace ignores entirely
(the gap is encoded in the next surface's `world_origin`).
3.7.8 fixes this properly: each trial copies the world surface
list with fresh `world_origin` / `world_R` arrays, then
perturbs the DOWNSTREAM surfaces' `world_origin` by
`delta_t * surface_i.world_R[:, 2]` for every positive-
thickness gap.  Radius perturbations remain per-surface (no
downstream shift).  Decenter still perturbs the ray bundle.

This is the correct world-frame analogue of the local-frame
thickness-chain perturbation: shifting the gap between surface
i and surface i+1 by `delta_t` shifts surface i+1 (and every
downstream surface including the image plane) by the same
delta along surface i's local +z direction, exactly as the
local trace would.

### 2D layout toolbar -- source-preview toggle exposed

The `show_source_preview` preference has existed since 3.6.1
but was undiscoverable (only settable by editing the prefs
dict directly).  Added a `Source preview` checkbox to the 2D
layout toolbar that persists to the same pref key, so users
can flick downstream preview rays on / off without leaving
the dock.

### 2D layout toolbar -- Export to SVG / PNG

New `Export…` button in the 2D layout toolbar opens a file
dialog and renders the current scene to either SVG (vector,
zoom-independent line weight, default) or PNG (raster, 2×
device-pixel-ratio so slide-deck inserts stay sharp).
Includes a 40-px margin around the bounding box matching the
"Reset View" framing.

### Wave-optics fold-awareness -- inventoried, deferred

The Wave Optics dock still routes through
`apply_real_lens_traced` (`lumenairy/elements/lenses.py`),
which builds its own surface list from the prescription
dict's `'surfaces'` key and calls the legacy local-frame
`trace()`.  The prescription's `'surfaces'` key drops
coord-break entries during `to_prescription()`, so wave-
optics propagation through folded systems silently runs the
unfolded equivalent path.  A `tilted_asm_propagate` exists in
`lumenairy.propagation` and could feed a fold-aware lens
propagator, but composing through a folded prescription
needs ~500-1000 lines of new core code (per-fold rotation
bookkeeping, tilted-frame Newton inversion in the OPL
extraction).  Documented as a known limitation; queued for a
future minor release.

### Non-sequential trace -- deferred to a major release

Beam splitters, ghost analysis, double-pass systems can't be
expressed in the sequential trace.  The 3.7.6 world-frame
surface representation (`world_origin` + `world_R` per
surface) is exactly the input a non-sequential trace would
need, so the foundation is in place -- but implementing the
ray-vs-surfaces priority search, the per-ray surface
candidate list, and the energy bookkeeping for
splitter / partial-reflection branches is the scope of an
architectural overhaul rather than a dock migration.  Queued
for the 3.9 / 4.0 line.

### Backwards compatibility

No public API removals.  All 26 validation suite files pass.

## [3.7.7] — 2026-05-11

World-frame correctness rollout to the remaining analysis docks,
3D undocked-shrink rescale fix, and a new GUI visual-regression
test that would have caught the 3.7.5 / 3.7.6 dock-shrink
regressions in 30 seconds.  Builds on the 3.7.6
`trace_world` infrastructure -- no new core library code,
just exposing `build_trace_surfaces_world()` / `build_run_trace
_world_surfaces()` as public methods and migrating four docks
to them.

### LumenAiry Designer is still in BETA

Layouts, dock arrangements, default workspaces, and even menu
names continue to change between minor releases.  Pin a specific
version if you depend on a particular GUI behaviour.

### Analysis docks — migrated to the world-frame trace path

These docks previously built their own surface list via
`SystemModel.build_trace_surfaces()` (legacy local list with
cb_pre Surfaces) and called `trace()` directly.  On folded
designs they inherited the same ~1/cos(θ) world-position
error that the 3.7.6 `run_trace` migration eliminated for the
2D / 3D layouts.

* **Footprint dock** -- now uses
  `build_run_trace_world_surfaces()` + `trace_world()`.  Per-
  surface footprint scatter plots correctly land on each
  post-fold lens's clear-aperture region in folded systems
  instead of being offset by ~1 mm.
* **Distortion dock** -- migrated chief-ray traces for both
  the 1D distortion sweep and the 2D distortion grid.  EFL /
  BFL still come from `system_abcd` on the legacy local list
  (paraxial calculation is identical between local and world
  lists for axially-symmetric systems; folded systems'
  distortion calculation now uses the world-correct chief-ray
  landing positions for the actual-vs-paraxial residual).
* **Spot vs Field dock** -- migrated; per-field spot diagrams
  now land at the correct world-frame image-plane positions
  through folded geometry.
* **Ray fan dock** (field-curvature plot only) -- the
  RMS-vs-field plot now uses `trace_world()`.  The ray-fan and
  OPD-fan plots still use `ray_fan_data()` / `opd_fan_data()`
  helpers on the legacy local list; those helpers are paraxial
  enough that folded-system inaccuracy is negligible at small
  field angles, and migrating them is a library-side change
  (queued for a future release).

### Not migrated (and why)

* **Tolerance dock** -- the Monte Carlo trial loop perturbs
  `Surface.thickness` field-by-field to model thickness
  tolerancing.  In the world-frame trace, `Surface.thickness`
  is unused (the gap to the next surface is encoded in the
  next surface's `world_origin`), so perturbing it would have
  no effect.  Proper migration requires perturbing the
  prescription's element distances before building world
  surfaces.  Stayed on the legacy path for this release; the
  thickness perturbation works correctly in the local trace
  even for folded designs (the world-position error is
  statistical and absorbs into the Monte Carlo spread).
* **Wave Optics dock** -- doesn't call `trace()` directly;
  routes to `apply_real_lens_traced` / `apply_real_lens` /
  `apply_real_lens_maslov`, which build their own surface
  lists from prescription dicts.  No surface-list migration
  needed.
* **Spot Diagram + MTF (analysis.py)** -- consume the model's
  pre-computed `_trace_result` (already world-frame from
  3.7.6) for spot RMS / Airy disk; only call `find_stop` /
  `system_abcd` paraxial helpers on the legacy list (which
  handles cb_pre surfaces correctly).  No migration needed.

### 3D layout -- pixel-fixed undocked shrink

Earlier 3.7.6 fixed the docked-shrink physics (the central
widget's `setMaximumHeight(350)` cap that pinned the splitter)
but the user noted that when the 3D layout is UNDOCKED and
the floating window is shrunk, VTK rescales the rendered
optics rather than clipping.  Fixed.

* `Layout3DView.eventFilter` intercepts the `QtInteractor`'s
  resize events and adjusts the camera's `parallel_scale` (or
  `view_angle` via tan, for perspective projection) by the
  ratio of new to old viewport height.  This keeps world-
  units-per-pixel constant -- the on-screen pixel size of
  every world feature stays fixed; the viewport just clips at
  its new edges.

### GUI visual-regression harness

New `validation/gui/test_layout_shrink.py` runs offscreen via
the existing `Harness` and is auto-discovered by
`validation/run_all.py`.  Self-skips with exit 0 when GUI
dependencies aren't installed (the workflow installs only
`[fft,perf,numba,hdf5,zarr,dev]`, NOT `[gui]`, so PySide6 /
matplotlib / pyvista aren't available on CI runners), keeping
the build green there while still exercising the GUI surface
locally for any developer with `pip install -e ".[gui]"`.
Covers:

* Layout2DView / Layout3DView construct on empty model
  without crashing.
* Both layouts accept `resize(60, 80)` (the dock-shrink
  smoke test).
* `MainWindow.layout_dock` starts at a sensible default
  (not pinned at minimum, not taking the whole window).
* `resizeDocks([200], Vertical)` actually shrinks the
  layout dock below its default.
* The layout dock's true minimum is < 80 px.
* tx71 chief-ray world-position smoke test:
  `SpatialFilter S1` lands at its vertex within 0.1 mm.

14 assertions; runs in ~4 seconds via `run_all`.

### Source rendering

Source glyphs in the 2D / 3D layouts (added in 3.6.1) are
still working: distinct glyphs per `source_type` (plane wave
bar, Gaussian ellipse, top-hat rectangle, fiber-mode annulus,
point-source dot, emitter-array grid).  Verified during this
release's smoke testing.

### Backwards compatibility

No public API removals.  `build_trace_surfaces()` continues to
return the legacy local-frame list for ABCD / paraxial /
tolerance / wave-optics consumers.  All 25 numerical
validation files pass unchanged; the suite now has 26 files
total with the new GUI test.

## [3.7.6] — 2026-05-11

Folded-design correctness pass plus a long-running dock-shrink fix.
Consolidates the 3.7.1 → 3.7.5 in-flight commits that fixed
specific folded-geometry issues into a single release built on
the new core `trace_world` engine (see `CHANGELOG.md`).

### LumenAiry Designer is still in BETA

Layouts, dock arrangements, default workspaces, and even menu
names continue to change between minor releases.  Pin a specific
version if you depend on a particular GUI behaviour.

### Folded designs — chief ray now lands on post-fold lens centres

* `SystemModel.run_trace` now uses the core's new world-coord
  trace path (`trace_world` + `_build_trace_surfaces_world`).
  Each surface in the trace list carries its absolute world
  position (`world_origin`) and orientation (`world_R`); rays
  propagate in world coords between surfaces.  Verified on
  tx4designstudy71: chief-ray world positions after a 45° fold
  mirror land at each subsequent lens centre to floating-point
  precision (`SpatialFilter`, `Lens 10`, `Lens 11`, `Lens 12`,
  `Detector` all hit dead-centre), and the chief-ray world
  direction after the fold is `(0, +1, 0)` — parallel to the
  post-fold optical axis instead of off by `1 - cos(45°)` ≈ 30 %.
* `surface_frames_2d_mm` / `surface_frames_3d_mm` no longer emit
  a separate cb_pre frame for tilted elements -- one frame per
  actual surface, matching the world trace's ray history
  one-to-one.  The 2D / 3D layouts read this and draw rays
  through each surface's exact world frame.
* Layout colour-coding now indexes off `trace_result.surfaces`
  (the actual traced list) instead of the legacy
  `build_trace_surfaces()` output, so direction-based mirror-
  count colours stay aligned with the ray history through any
  fold geometry.
* Recovered from earlier 3.7.x experiments (rolled into this
  release): 3D fold orientation matches 2D (3.7.1), cb fields
  no longer stripped in the run_trace fresh-copy (3.7.2),
  `Element.origin` / `Element.R` cached as canonical world
  frame so the prescription editor's absolute-coords mode
  actually edits absolute (3.7.3), Zemax → world conversion is
  now exact for folded designs (3.7.4), and ray segments are
  coloured by direction (mirror count) in both 2D and 3D so
  forward / return / post-fold paths visually separate (3.7.5).

### Layout-dock shrink — root cause finally found

Multiple previous attempts (`setMinimumSize(0, 0)`,
`minimumSizeHint(40, 40)` override, `setSizeConstraint
(SetNoConstraint)`, `QDockWidget.setMinimumSize(0, 0)`) failed
because they addressed the wrong constraint.  The real culprit
was `self.element_editor.setMaximumHeight(350)` on the central
widget: when the user dragged the splitter UP to shrink the
2D / 3D layout dock, the freed vertical space wanted to flow
into the central widget -- but it was capped at 350 px, so the
freed space had nowhere to go and Qt refused to move the
splitter at all.

* Removed `element_editor.setMaximumHeight(350)`.  The element
  table now grows naturally into freed vertical space; its
  internal scrollbar handles overflow.  Verified
  programmatically: `resizeDocks([target], Vertical)` now
  actually resizes the layout dock to any value between its 56
  px minimum and the available area.
* Reverted the over-aggressive `QSizePolicy.Ignored` policies
  I'd added to the inner `QGraphicsView` / `QtInteractor` /
  toolbar in the prior round -- those made the docks start at
  minimum size on first launch (the opposite problem) because
  `Ignored` tells Qt the widget doesn't want any space.
  Restored to `Expanding` so docks start at a useful default.
* 3D layout's button row is now a `QToolBar` (was a
  `QHBoxLayout` of `QPushButton` with hard `setMinimumWidth`).
  Qt's `QToolBar` has built-in overflow handling -- when the
  dock is narrower than the natural action row, hidden buttons
  go into a `">>"` popup instead of forcing the dock wide.
* `Layout2DView.resizeEvent` no longer calls `fitInView` --
  shrinking the dock now CLIPS the visible scene portion (and
  shows scrollbars if needed) instead of rescaling the entire
  drawing.  Users complained the rescale-on-resize destroyed
  their chosen zoom and made the layout feel "live" in a
  distracting way.

### View ▸ Reset to Default Layout

* New top-level **View ▸ Reset to Default Layout** menu action.
  Captures the freshly-built default dock geometry once on
  startup (after `_create_docks()`, before any persisted
  workspace state restoration) and lets the user restore it
  on demand without restarting the application.  Un-floats /
  re-tabs / re-sizes every dock back to first-launch state and
  resets the workspace set to the built-in defaults.
  Prescription, undo history, and preferences are unaffected.
* The buried **View ▸ Workspace ▸ Reset Workspaces to Defaults**
  entry remains for users who want to reset only the workspace
  definitions without touching dock geometry.

### Backwards compatibility

* `lumenairy.raytrace.trace()` (the legacy local-frame trace)
  is unchanged.  Custom Surface lists, ABCD / paraxial /
  Seidel helpers, the wave-optics dock, tolerance / rayfan /
  spot-field / distortion / footprint docks (all of which
  build their own surface lists and call `trace()` directly)
  continue to work without modification.
* `Surface` gains two optional fields (`world_origin`,
  `world_R`, both default `None`).  Existing constructor calls
  and serialised systems are unaffected.
* All 25 validation suite files pass unchanged.

## [3.7.0] — 2026-05-10

Tilt-aware ray tracing reaches the GUI, plus the folded-geometry
layout hotfixes (3.6.1 hotfix-4 → -6) consolidated into a single
release.  Imported `.zmx` prescriptions with COORDBRK surfaces
now render folded **and** the rays actually deflect at the fold;
the round-trip through Zemax export/import preserves mirrors
and coord-breaks.

### LumenAiry Designer is still in BETA

Layouts, dock arrangements, default workspaces, and even menu
names continue to change between minor releases.  Pin a specific
version if you depend on a particular GUI behaviour.

### Layout views — fold geometry + ray tracing

* `SystemModel._build_trace_surfaces_internal()` emits a
  coord-break Surface immediately before any element with
  non-zero `tilt_x` / `tilt_y` / `decenter_x` / `decenter_y`,
  so the core trace's new cb handler (3.7.0 core release)
  actually folds rays at the right places.
* New `surface_frames_2d_mm()` / `surface_frames_3d_mm()`
  helpers emit a per-traced-surface frame in trace order —
  including a frame entry for each cb so `_draw_rays` stays
  index-aligned with the new trace history length.
* `element_frames_2d_mm()` / `element_frames_3d_mm()` updated
  to compute world positions in physical-positive convention
  (post-mirror distances no longer double-count the Zemax
  signed-thickness convention).
* `to_prescription()` now emits `elements` + `all_thicknesses`
  + `coord_breaks` for round-trip preservation; the legacy
  lens-only `surfaces` / `thicknesses` keys remain.

### Layout dock UX

* Layout dock can finally be dragged down to a small height
  to give the prescription editor more vertical room.
  Required overriding `minimumSizeHint()` (the inherited
  implementation walked the QGraphicsView size hint and
  reserved several hundred pixels) plus `QSizePolicy.Ignored`
  on both Layout2D and Layout3D.
* Wheel-zoom now always zooms — even at high zoom where the
  QGraphicsView would previously route the wheel to its
  scrollbars instead of bubbling up to our handler.
* New "Reset View" button on the 2D layout toolbar (3D
  layout already had one).
* New ⟲ / ⟳ in-plane rotation buttons on the 3D layout
  toolbar (camera roll about the forward axis).

### Prescription import

* `.zmx` and `.txt` prescription import now consumes the full
  importer-side `elements` list (with mirrors and surf_num)
  plus `all_thicknesses` and `coord_breaks`, instead of the
  lens-only filtered `surfaces` list.  Mirrors no longer
  vanish on import; coord-breaks land on the right element.
* Element labels prefer the per-surface `COMM` field
  ("Diffractive", "SpatialFilter", "Metasurface" in
  tx4designstudy71.zmx) and fall back to running "Lens 1" /
  "Mirror 1" tags instead of repeating the source filename
  on every row.
* Dummy air-air infinite-radius surfaces with no comment are
  silently skipped and their thicknesses rolled into the
  next rendered element's distance.  Specialty air-air
  planes that *do* carry a comment get rendered as flat
  zero-thickness elements labelled by the comment.

### Known limitations

* The core ray tracer now folds rays at coord-breaks, but
  the **layout view's ray rendering** can still look off
  for some folded designs depending on local-frame vs.
  world-frame conventions in the cb's neighbourhood.
  Refinement continues; please file specific examples
  (screenshots + the prescription) to the wiki Issues page.
* Wave-optics propagation does not yet honour fold
  geometry; Wave Optics dock results on a folded system are
  still computed against the un-folded equivalent.
* Detector position from BFL still uses the un-folded
  paraxial calc.  Workaround: edit the detector distance
  manually after import.

## [3.6.1] — 2026-05-10

GUI overhaul addressing the 3.6.0 audit's four headline issues
plus three hotfixes against bugs the user caught after the first
3.6.1 push, plus the removal of the legacy `optical-designer`
launcher / console-script aliases that 3.5.9 had retained for
backward compatibility.

### Initial 3.6.1 push

Stages A + B + C of the 3.6.0-audit overhaul:

* **Stage A — layout-bug fixes.**  Sources are now drawn in 2D
  and 3D layouts as per-source-type glyphs (bar / ellipse /
  disc / annulus / dot / array).  Source-type combo changes
  refresh the layouts immediately (no 200 ms debounce wait).
  Selecting a row in the prescription editor highlights the
  corresponding element in 2D (amber rectangle) and 3D (gold
  ring); highlight survives auto-retraces.
* **Stage B — workspace UX cleanup.**  `DEFAULT_WORKSPACES`
  trimmed (Analysis 13 → 5, Wave Optics 9 → 4, others
  rebalanced).  New top-level "View > Configure Workspace
  Docks…" action.  All 3.6.0 specialty docks now exposed in
  the View menu's flat toggle list.  One-time "Workspaces
  simplified — reset?" migration prompt for upgraders.
* **Stage C — industry-pattern adoptions.**  Optional source-
  preview rays in the 2D layout (Optiland / Zemax pattern).
  OSLO-style "Attach slider to this parameter…" right-click
  action on the surface sub-table.

### Hotfix-1 (bundled into 3.6.1)

* `layout3d` was dropped from the Design workspace defaults in
  the Stage B trim.  Restored.  `DEFAULTS_REVISION` bumped 5 → 6
  so existing users with persisted layouts get layout3d added
  back via the union-merge migration path.
* The Stage C source-preview rays were drawn UPSTREAM of the
  source plane and read as converging-into-the-source for non-
  plane-wave types.  Rewrote them to draw DOWNSTREAM (source
  plane → first lens surface) with per-source-type patterns
  (parallel for plane-wave, diverging from waist for Gaussian,
  cone with NA-derived half-angle for fiber-mode, diverging fan
  from the source point for point-source).  Default flipped to
  OFF so the overlay is opt-in.
* Detector wasn't selectable from the 2D layout.  Registered a
  click zone for the detector at the image-plane line so it
  matches the source / lens click behaviour.

### Hotfix-2 (bundled into 3.6.1)

* Both `_draw_rays` paths (2D and 3D) built `z_positions` from
  the cumulative thicknesses of the optical surfaces only, via
  `build_trace_surfaces`.  That ignores the source-to-first-
  surface air gap, so rays for a typical 50-mm-source-to-lens
  + 100-mm-detector setup were squished into the first ~10 mm
  of the layout instead of spanning the actual ~150 mm system
  length.  Plus no segment was ever drawn from the source
  plane to the first lens surface, so the parallel pre-lens
  beam was invisible.
* Rewrote both `_draw_rays` to compute world-frame z for every
  traced surface using `element_z_positions_mm()` plus per-
  surface offsets within each element, append the detector z
  for the final history bundle, and prepend a (z=0, y_in) →
  (z=z_first_surf, y_in) segment so the parallel-beam pre-lens
  leg is visible.  Verified: chief ray (input y=0) now reaches
  z=150 with y=0; marginal ray (input y=−2.21 mm) stays
  parallel from z=0 to z=50, refracts, and converges to
  y=0.55 mm at the image plane (close to focus, slight
  spherical aberration as expected).
* (A self-introduced unit bug in the first cut of this fix
  that multiplied an already-mm thickness by 1e3 was caught
  and corrected before commit.  Surface thicknesses on the
  GUI-side `Element.surfaces` are in mm; the metres-scale
  appears only on the core-library `Surface` objects from
  `build_trace_surfaces`.)

### Hotfix-3 (bundled into 3.6.1)

* The 3D layout reset its camera orientation on every
  `system_changed` rebuild, so any parameter edit (radius
  bump, source-type change, distance edit) snapped the user
  back to the iso default and undid their orbit / zoom / pan.
* Fixed by snapshotting `self._plotter.camera_position` BEFORE
  `plotter.clear()` and restoring it after the new scene has
  been drawn.  Only the very first rebuild snaps to iso (and
  flips a `_camera_initialized` flag); subsequent rebuilds
  preserve the saved camera.  The toolbar's "Reset View"
  button still calls `_reset_camera()` directly, so the user
  can deliberately recenter when they want to.

### Removed: legacy launcher + script alias

* `run_optical_designer.py` deleted.
* `optical-designer` console-script entry removed from
  `pyproject.toml`.
* `run_lumenairy_designer.py` no longer imports from the old
  launcher; the `main()` function is now inlined and uses the
  top-level `lumenairy as la` namespace for the prescription
  helpers (`thorlabs_lens`, `load_zmx_prescription`,
  `load_zemax_prescription_txt`) -- the original launcher had
  imported them from a non-existent `lumenairy.prescriptions`
  submodule, an IDE-flagged bug that worked only because the
  fallback path hit when `--demo` wasn't supplied.

The QSettings key `(lumenairy, OpticalDesigner)` is unchanged
so existing saved workspaces, recent files, and pinned-dock
state survive the launcher / alias removal.

### Layout views — sources are now drawn

`Layout2DView._draw_source` and `Layout3DView._draw_source_3d`
render the optical source as a per-source-type glyph at z=0:

* **Plane wave**: vertical bar across the EPD (2D) / disc cap
  perpendicular to the optical axis (3D), plus propagation-
  direction arrows / cone glyph.
* **Gaussian**: filled ellipse sized by `beam_diameter_mm`
  (2D) / oblate sphere (3D).
* **Gaussian aperture**: ellipse sized by `sigma_mm`.
* **Top-hat**: hard-edge rectangle / disc of width
  `top_hat_diameter_mm`.
* **Fiber mode**: annular ring outline with MFD-derived inner
  radius.
* **Point source**: small dot.
* **Emitter array**: grid of dots at the configured pitch
  (capped at 7×7 visible).

Source `.describe()` text is shown above the glyph in mint
green to match the source-row tint in the element table.  The
source z-position is registered into `_surface_zones` so a click
in the layout selects element 0 (the source row).

### Layout views — bidirectional table-layout selection highlight

Previously the layout-click → table direction worked but the
table-row-click → layout direction was missing.  3.6.1 adds:

* `Layout2DView.set_selected_element(idx)` and
  `Layout3DView.set_selected_element(idx)` slots.
* `_redraw_highlight()` / `_redraw_highlight_3d()` overlays a
  translucent amber rectangle (2D) or gold ring (3D) on the
  selected element's z-zone.  Highlight survives auto-retraces
  (re-added at the end of `rebuild()`).
* `ElementTableEditor.element_selected_in_table = Signal(int)`
  emitted from `_on_row_selected()`.
* `MainWindow.__init__` wires the signal into both layout views.

### Source-type changes refresh layouts immediately

Previously `_on_source_type_changed` only triggered a layout
rebuild after the 200 ms text-input debounce timer expired.
3.6.1 stops the timer and applies the new source synchronously
on the discrete combo-box change, so the 2D / 3D glyph updates
immediately.

### Workspace defaults trimmed

`DEFAULT_WORKSPACES` rebalanced to focused minimal sets:

| Workspace    | 3.6.0 | 3.6.1 | Default docks                              |
|--------------|-------|-------|--------------------------------------------|
| Design       | 5     | 3     | layout, library, summary                   |
| Optimize     | 6     | 3     | layout, optimizer, sliders                 |
| Analysis     | **13**| 5     | layout, spot, rayfan, summary, psfmtf      |
| Wave Optics  | **9** | 4     | layout, waveoptics, zernike, interferometry |
| Tolerancing  | 4     | 3     | layout, tolerance, sensitivity             |
| Materials    | 4     | 2     | materials, glassmap                        |

`DEFAULTS_REVISION` bumped 4 → 5.  Existing users with
persisted layouts get a one-time "Workspaces simplified — reset
to new defaults?" prompt on first 3.6.1 launch via the new
`WorkspaceManager.needs_reset_prompt` signal; choosing No
preserves their existing layout.  The new docks (caustic,
richards_wolf, coherence, shack_hartmann, lg_aberration, rcwa)
remain available via View > Configure Workspace Docks…

### View menu — Configure Workspace Docks promoted; specialty docks listed

* New top-level **View > Configure Workspace Docks…** action
  (was previously buried two levels deep under View >
  Workspace > Manage Docks).
* The View menu's flat dock-toggle list now exposes every 3.6
  specialty dock (caustic, richards_wolf, coherence,
  shack_hartmann, lg_aberration, rcwa) and the welcome / repl /
  diagnostics docks, organised into four groups separated by
  separators: Layouts & overview / Geometric analysis / Wave
  optics & specialty / Materials, optimization, utilities.

### Source-preview rays in the 2D layout

`_draw_source` now also draws short upstream preview rays
illustrating propagation direction.  Pattern depends on source
type (parallel rays for plane-wave, converging fan for
Gaussian, diverging fan from a virtual point for point-source,
etc.).  Coloured by wavelength to read as a visual extension of
the downstream traced rays.  Toggleable via the new
`SystemModel.prefs['show_source_preview']` flag (default ON).

### OSLO-style "Attach slider to this parameter…"

Right-click any numeric cell in the surface sub-table (Radius,
Thickness, Semi-Diam, Conic, Radius Y, Conic Y) → "Attach
slider to this <field>…".  In one click:

* `SystemModel.add_optimization_variable(elem_idx, surf_idx,
  field)` appends to `opt_variables` and emits
  `system_changed`.
* The Sliders dock is raised automatically.
* The corresponding `ParameterSlider` is generated (idempotent
  if already present) and pulses amber for 3 seconds so the
  user can see which one is new.
* Drag it → 2D layout's existing 200 ms debounce kicks in →
  retrace + redraw.

This collapses the previous multi-step path (open Optimizer
dock → variable-grid dialog → tick checkbox → OK → switch to
Sliders dock → click Generate → drag) into a single right-click.

### Internals

* `SurfaceDetailPanel` gained a `slider_attach_requested` signal
  forwarded out via `ElementTableEditor.slider_attach_requested`
  to `MainWindow._on_slider_attach`.
* `WorkspaceManager.load_json` defers `needs_reset_prompt`
  emission via `QTimer.singleShot(0, ...)` so the modal pops
  *after* the caller finishes wiring (avoiding re-entrancy
  with toolbar/menu construction).

### Backwards compatibility

* No public-API changes.
* Existing user customisations (saved JSON designs,
  workspaces, recent files, dock layouts, pinned docks) all
  survive the upgrade.  QSettings storage key unchanged.

## [3.6.0] — 2026-05-09

GUI feature-coverage release.  Closes ~30 of the 42 audit findings
identified after the 3.5.9 release: workspace gaps, menu wiring,
new specialty docks, dispatch additions, theme cleanup, source-
factory expansion, optimizer hierarchy, run-button consistency,
keyboard shortcuts, what's-new in-app modal, expanded Help and
Tools menus, welcome-dock redesign, REPL improvements.

### Five new specialty docks

- **Richards-Wolf focus** (`richards_wolf_dock.py`): vector-
  diffraction PSF for high-NA imaging, plotting |Ex|² / |Ey|² /
  |Ez|² components plus total intensity.  Chooses pupil
  polarization from a 6-option preset list.
- **Partial coherence (Köhler)** (`coherence_dock.py`): Köhler-
  decomposed extended-source imaging with a σ slider and circular
  / annular / dipole / quadrupole illumination shapes.
- **Shack-Hartmann sensing** (`shack_hartmann_dock.py`): virtual
  microlens-array wavefront sensor on the most recent wave-optics
  focal field.  Plots per-lenslet slope magnitude and a
  reconstructed Zernike spectrum (auto-fed when the Wave Optics
  dock finishes).
- **LG aberration tensor** (`lg_aberration_dock.py`): heat-map of
  the Laguerre-Gauss aberration tensor with Seidel-equivalent
  labels for the largest elements.
- **RCWA grating** (`rcwa_dock.py`): rigorous-coupled-wave 1-D
  grating solver with groove-profile / period / depth / duty-cycle
  / polarization controls and a wavelength-sweep efficiency plot.

All five live in the appropriate workspace by default
(Analysis or Wave Optics) via `DEFAULT_WORKSPACES` and have menu
entries in the Analysis menu.  Workspace defaults_revision bumped
3 → 4; the migration logic now correctly union-merges new docks
into existing workspaces (closing the 3.5.9 caustic-dock gap).

### Wave Optics dispatch — four new whole-prescription propagators

The Method dropdown gained `GBD`, `HFPI`, `Huygens-Fresnel`, and
`Subaperture` entries.  When one of these is selected the worker
short-circuits the per-element loop and calls the corresponding
`propagate_*_through_prescription` from the core library on the
full prescription dict, returning a single focal-plane field.

### Wave Optics — Quick-run presets, Detector model

- **Quick run** preset bar at the top of the dock with three
  buttons (Fast preview / Production / Sub-nm validation), each
  writing a complete config (N / dx / method / lens model /
  precision / bandlimit) so a new user is one click away from Run.
- **Detector model** group exposes `apply_detector` as an opt-in
  post-processing step: pixel pitch, QE, read noise, dark current,
  exposure time.

### Source factories — top_hat, fiber_mode, source polarization

- `SourceDefinition.TYPES` now includes `top_hat` and `fiber_mode`;
  `to_source()` routes to `Source.top_hat()` / `Source.fiber_mode()`.
- Element table source-row form has the corresponding parameter
  rows (top-hat diameter, fiber MFD + NA).
- New `polarization` field on `SourceDefinition` (None /
  linear_x / linear_y / linear_45 / rcp / lcp) for future Jones-
  field plumbing.
- Insert > Source submenu offers six one-click presets.

### Application-shell changes

- **Keyboard Shortcuts dialog** is now a sortable + filterable
  `QTableWidget` covering all ~25 shortcuts (was a 13-line plain-
  text `QMessageBox`).  Auto-derived from a single source-of-
  truth list so it never drifts.
- **What's New in 3.6.0** in-app modal, triggered automatically
  on first launch after a version bump (`QSettings('lumenairy',
  'OpticalDesigner').value('last_seen_version')` comparison).
  Manually re-openable via Help > What's New.
- **About dialog** now lists detected optional dependencies
  (pyFFTW / CuPy / JAX / h5py / zarr / numba / astropy) so users
  know what backends are active without opening Wave Optics.
- **Help menu** expanded: Wiki / GUI README / Examples folder /
  Open Demo / What's New / Report a Bug.
- **Tools menu** expanded (Scale system, Find nearest Thorlabs,
  Quick Zernikes from trace, Chromatic focal shift).
- **F-key shortcuts** for one-keystroke analyses: F5 wave optics
  (existing), F6 retrace, F7 through-focus, F8 Zernike,
  F9 PSF/MTF, F10 caustic.  Hint shown in each dock's Run-button
  text.  F-key dispatcher (`_fkey_run`) raises the named dock and
  triggers its Run-equivalent action.
- **Save As** (Ctrl+Shift+S) added to the File menu.
- **Status-bar metrics** (EFL / BFL / f# / EPD / λ) now clickable
  → raise System Data dock.

### Welcome dock redesign

- Hero **Open Demo (AC254-100-C)** button at +2pt size with the
  primary-action stylesheet, separated from the secondary row.
- Secondary row now includes an **Open Python REPL** button +
  reworded subtitle that mentions drag-drop and the REPL.
- Subtitle calls out the `model.load_prescription(rx)` REPL flow
  for users coming from Python scripts.

### REPL improvements

- Banner expanded with five quick examples covering the most
  common workflows (load Thorlabs lens, push into model, save to
  user library, plot last wave-optics PSF).
- `lumenairy` is now pre-imported as `la` in the REPL globals,
  matching the convention used in library examples.
- Dock title ready to be renamed "Python (REPL)" via the
  default_dock_titles map.

### Optimizer redesign

- Three coequal Local / Global / Wave buttons replaced with a
  single primary **Optimize** button (which auto-routes to the
  hybrid wave engine when a wave merit is selected) plus a
  secondary disclosure row for **Global Search…** / **Wave
  Optimize…** advanced modes.
- JAX wave-propagator checkbox moved into a dedicated **Compute
  backend** group at the top of the Optimization group, paralleling
  the Wave Optics dock's Compute group.

### Run-button standardisation

Every dock's primary action now sets
`objectName('run_button')`.  A single new stylesheet rule in
`apply_theme` styles all run buttons identically (accent colour
background, bold weight, larger padding).  Where the dock had a
non-standard label ("Run", "Compute Jones pupil", "Rank
variables", "Jointly optimise"), the label is now "▶ Run <noun>"
or similarly verb-glyph-noun for consistency.  F-key hints
appended where applicable.

### Workspace migration fix

`workspace.py:534-560` (load_json) now union-merges new default
docks into existing workspaces of the same name, not just adding
new workspaces.  Closes the bug where the 3.5.9 caustic dock
silently disappeared for users with persisted layouts.

### Tolerance dock — limitations surfaced

- Yellow-bordered banner at the top of the dock explaining the
  ray-trace MC limitations.
- Export Report JSON schema gained a `limitations` field
  enumerating the same caveats so downstream tooling can detect
  them.
- Disabled Export-Report button now has a tooltip explaining
  what's needed to enable it.

### Internals

- New `_fkey_run(dock_attr)` dispatcher introspects each dock for
  a `btn_run` / `btn_compute` / `btn_recompute` attribute (or a
  `run` / `compute` / `recompute` / `_run` / `_compute` method)
  to support one-keystroke runs without per-dock special cases.
- New `_apply_quick_preset(key)` method on the Wave Optics dock
  writes complete per-preset configs through the existing widget
  setters, so the run path is unchanged.
- New `_show_whats_new(force_show)` + `_maybe_show_whats_new_on_startup`
  on MainWindow for the in-app changelog modal.
- New `_ins_source_preset(kind)` helper for the Insert > Source
  submenu.

### Backwards compatibility

No public API additions or removals on the GUI side.  Existing
saved JSON designs / workspaces / pinned docks / recent files
all survive.  Top-level `__init__.py` exports unchanged.

## [3.5.9] — 2026-05-09

GUI catch-up to the 3.3-3.5 core-library work, plus an application
rename.  Closes the largest user-visible gap since 3.2.14: every
3.5.7 propagator and 3.5.4 analysis utility now has a UI surface,
and the optimizer + tolerance docks gain new integration points
for the JAX path and structured reports.

### Application rename — Optical Designer → LumenAiry Designer

- Window title, About dialog, welcome panel, file-dialog filters,
  and `app.setApplicationName` now read **LumenAiry Designer**.
- New `run_lumenairy_designer.py` launcher and
  `lumenairy-designer` console-script entry point in
  `pyproject.toml`.
- `run_optical_designer.py` and the `optical-designer` script are
  kept as backward-compatible aliases for users with existing
  shortcuts / CI scripts; both invoke the same `_cli_main` entry
  point.
- The `QSettings('lumenairy', 'OpticalDesigner')` storage key is
  intentionally **left unchanged** so user customisations
  (workspaces, recent files, dock layouts, pinned docks) survive
  the upgrade without a migration step.

### Wave Optics dock — propagator family + post-processing

Wired the 3.5.7-3.5.8 propagator additions through the
`combo_method` dispatch and the `_run` config dict:

- **Three new MFT methods** in the Method dropdown: *Fresnel
  MFT*, *Fraunhofer MFT*, *ASM MFT*.  When an MFT method is
  selected, a *Focal-plane zoom* parameter group is revealed so
  the user can set `dx_out` (µm), `N_out`, and `centre_out` (x,
  y).  Between-element steps fall back to the corresponding non-
  MFT base method on the natural grid; only the to-focus step
  uses the Bluestein chirp-Z path.  Standard tool for focal-
  plane zoom (sample a tightly-focused region at sub-FFT-pitch
  resolution without padding the input).
- **Bandlimit (Matsushima)** checkbox.  Single dock-wide flag
  passed to ASM, RS, and ASM-MFT calls (between-element + to-
  focus).  Default ON.  Surfaces the kwarg added to
  `rayleigh_sommerfeld_propagate` in 3.5.8.
- **Convert focal field to chief-relative OPD (R = v − f)**
  checkbox.  When enabled, applies `apply_fresnel_curvature`
  (3.5.7) to the focal-plane field with `R = bfl − efl` so the
  output can be compared bit-for-bit against ray-trace-rooted
  aberration tools (OPDPy, Zemax OPD operands).
- **Recommend grid** button now delegates to
  `la.recommend_grid_for_prescription` (3.3.3) when the system
  exports as a prescription dict; falls back to the local NA-
  based heuristic otherwise.  Picks up DOE diffraction-order
  spread handling and any future core-side recommendation
  improvements automatically.

### Wave Optics dock — Custom MHS chain tab

New tab inside the Wave Optics dock for advanced users who want
to drive `MhsPipeline` (3.5.0) directly:

- Method selector for per-subdomain propagator (`gbd` / `asm` /
  `fresnel` / `rayleigh_sommerfeld`).
- Pre- and post-distance fields for free-space sections before
  the first refractive surface and after the last.
- **Build pipeline** constructs `MhsPipeline.from_prescription`
  from the current model + the per-element tab's `N` / `dx` /
  wavelength.
- Subdomain inventory table (in z, out z, label) so the user can
  inspect the constructed plane layout before pressing Run.
- **Run pipeline** propagates the source field through every
  subdomain, with a per-plane peak / label summary.

### Tolerance dock — Export Report

- New **Export Report…** button writes a structured JSON or text
  report from the cached MC results.  Schema is versioned
  (`'kind': 'lumenairy_tolerance_report', 'version': '1'`) so
  downstream consumers can rely on the structure.  Includes per-
  knob tolerances, summary stats (mean / std / median / p05 /
  p95 / min / max for both RMS spot and EFL), and the full per-
  trial arrays.
- A future Strehl-based MC + JAX backend integration via
  `tolerancing_report` and `monte_carlo_tolerancing_jax` is
  deferred to a follow-up release: the existing dock's
  perturbation model (radius-%, thickness-mm, decenter-mm)
  doesn't directly map to the core's decenter / tilt / form-
  error MC API, so a clean integration deserves its own design
  pass.

### Optimizer dock — JAX wave propagator toggle

- New **Use JAX wave propagator** checkbox.  When checked, passes
  `wave_propagator='real_lens_traced_jax'` into `design_optimize`
  so the optimizer's default `jac='auto'` strategy can build
  analytic Jacobians (via `jax.grad`) for any JAX-aware merit
  terms.  Falls back to FD for non-JAX merits.  Greyed out when
  `jax` is not installed.

### New: Caustic Diagnostic dock

- Wraps `caustic_diagnostic` + `plot_caustic_diagnostic` (3.5.4).
- Inputs: fan radius (µm), samples per gap, optional post-surface
  sample length (mm).
- Output: matplotlib plot (delegated to
  `plot_caustic_diagnostic` for visual consistency with the
  library's reference rendering) + a text summary listing
  detected caustic crossings with their Maslov indices.
- Runs in a background thread to keep the UI responsive on
  multi-element systems.
- Lives in the **Analysis** workspace by default
  (`defaults_revision` bumped 2 → 3 so existing saved workspaces
  pick it up without losing customisations).

### New: Tools > Scale system…

- New menu item under a new **Tools** menu in the menu bar.
- Uses `scale_prescription` (3.3.3) to multiply every linear
  dimension (radii, thicknesses, semi-diameters, aspheric
  coefficients) of the current system by a user-specified factor.
- F-number, NA, and paraxial magnification are preserved by
  geometric self-similarity; wavelength is NOT scaled.  Useful
  for unit conversions (mm ↔ m) and for cheaper polynomial-fit-
  based diffraction methods at smaller absolute extents.

### `SystemModel.SourceDefinition` ↔ `lumenairy.Source` integration

- New `SourceDefinition.to_source(N, dx_m, epd_m=...)` method
  returns a `lumenairy.Source` (3.5.0) instance built via the
  appropriate core factory (`Source.gaussian`,
  `Source.plane_wave`, `Source.point_source`).  ``emitter_array``
  -- which has no 1:1 core factory -- builds a tiled superposition
  of ``Source.gaussian`` instances and wraps them in a `Source`
  via the dataclass constructor.
- Wave Optics dock's `_run` now uses `to_source()` to construct
  the source field whenever possible, so the dock's source path
  stays in lockstep with the rest of the library's source story.
  Falls back to the legacy hand-rolled construction if
  `to_source()` raises.

### `PropagationResult` opt-in

- `WaveOpticsWorker.run` now also produces a
  :class:`lumenairy.PropagationResult` (3.5.0) and emits it in
  the result dict under the new key ``propagation_result``.
  Existing keys (``planes``, ``I_focus``, ``dx``, ``power_in``,
  etc.) are unchanged so nothing breaks; downstream consumers
  can opt into the unified wrapper by reading
  ``result['propagation_result']``.
- The wrapper carries the focal-plane field, output dx,
  wavelength, and method as the canonical fields, plus a
  ``history`` list of per-plane (label, field, z) tuples and
  a ``metadata`` dict with the forecast inputs (lens model,
  bandlimit, MFT params, chief-relative-OPD flag, power-in /
  power-focus / d4sigma / elapsed time).

### Internals / structure

- Wave Optics dock is now a `QTabWidget` with two tabs:
  *Per-element propagation* (the existing flow) and *Custom MHS
  chain* (new).  All existing widgets, signals, and the
  `run_finished` signal are unchanged on the per-element tab.
- `lumenairy/ui/caustic_dock.py` (new file, ~250 LOC) follows the
  existing dock conventions (matplotlib canvas, `QThread`
  worker, `_draw_empty` / `_draw_result` helpers).
- `_current_method_key()` factored out as the single source of
  truth for the wave-optics method-key conversion (used by both
  the forecast and the run dispatch).

### Deferred to a future release

- Strehl-based MC integration in the tolerance dock (the dock's
  existing perturbation model -- radius-%, thickness-mm,
  decenter-mm -- doesn't directly map to the core's decenter /
  tilt / form-error MC API; a clean integration deserves its own
  focused design pass).

## [3.2.14] — 2026-04-24

### Performance perceptible in the GUI

Mirrors the core-library 3.2.14 perf pass.  The GUI does not change;
typical workflows are simply faster:

- Multi-config / wavelength sweeps + optimization loops now hit the
  ASM transfer-function `H` cache when the geometry repeats — ~1.5×
  speedup per ASM call on N=2048 grids.
- `JonesField.propagate` (used by the Wave Optics dock when the
  source is polarized) runs Ex/Ey through a single batched FFT pair
  on grids ≥ 512.
- Single-precision (`np.complex64`) toggle now exposed at the
  package top level — flip once for ~2× FFT throughput and ~2× more
  headroom on memory-tight grids.
- Aspheric-surface sag computation is now numba-fused (one threaded
  pass over the grid, no per-coefficient temporaries).  ~4.75×
  speedup on N=2048 with 5 aspheric coefficients.

## [3.2.13] — 2026-04-24

### No GUI-side changes

Validation-suite expansion in the core (~70 new physics / interop
test cases, 298 total assertions across 16 files, all PASS).  The
GUI inherits the safer regression net but has no user-facing change.

## [3.2.12] — 2026-04-24

### UI quality-of-life: keyboard, drag-drop, REPL, compact mode

- **`Ctrl+1` … `Ctrl+9`** — jump directly to workspace tab N.
- **Window title** shows current file + dirty marker (`Optical
  Designer — file.zmx*`).
- **Drag-and-drop** any `.zmx` / `.txt` / `.seq` / `.json` onto
  the window to load.
- **Permanent right-aligned status-bar metrics**: EFL, BFL, f/#,
  EPD, λ — visible on every workspace.
- **Pinned docks** across all workspaces (`View > Workspace > Pin
  Docks Across All Workspaces…`).
- **Welcome dock** — empty-state landing panel with quick-start
  buttons + recent-files list backed by `QSettings`.
- **Embedded Python REPL** dock with `model`, `np`, `plt`,
  `result`, `wave` pre-bound to the current system + latest run.
- **Workspace export/import** as `.workspace` JSON files for
  sharing custom layouts.
- **Optimizer progress badge** on the Optimize tab title
  (`Optimize • iter N`) while running.
- **Element-table polish**: right-click context menu, amber
  highlight on cells with optimization variables, search box.
- **F11 / Compact Mode** — hides menu bar + dock title bars.

## [3.2.11] — 2026-04-24

### Workspace defaults rebalanced

- Added a dedicated **Optimize** tab between Design and Analysis
  holding Optimizer + Sliders + Multi-Config + Snapshots.
- Slimmed the **Design** tab to just 2D / 3D Layout + System Data
  + Library — the docks you actually look at while building a
  layout.
- Dropped Jones Pupil from the Wave Optics tab defaults (still
  available via `View > Jones Pupil` or Manage Docks).
- Added a `defaults_revision` migration so existing users with
  saved layouts pick up the new tabs without losing customisations.

## [3.2.10] — 2026-04-24

### Top-of-window workspace tabs

A tabbed-workspace system reduces GUI clutter by grouping the 27+
analysis docks by topic.  Each tab shows only the docks relevant
to that phase of design work.

- New `ui/workspace.py` with `Workspace`, `WorkspaceBar`,
  `ManageWorkspaceDialog`, `WorkspaceManager`.
- Default workspaces: **Design**, **Analysis**, **Wave Optics**,
  **Tolerancing**, **Materials**.  Right-click any tab for Manage
  Docks / Rename / Duplicate / Delete; `+` button to add new
  workspaces; double-click to rename.
- Per-tab dock geometry preserved on switch via
  `QMainWindow.saveState()` / `restoreState()`.
- User-initiated dock visibility changes (close button, View menu
  toggle) automatically update the active workspace's dock list.
- Persistence to `QSettings('lumenairy', 'OpticalDesigner')` —
  custom workspaces survive restart.
- Wired into `main_window.py` with `_build_workspace_bar()` +
  `_init_workspaces()` + a `View > Workspace` submenu (with Reset
  to Defaults).

## [3.1.6] — 2026-04-21

### No GUI-side changes

- Core-library reliability fix: zarr storage writes now succeed on
  Windows + Python 3.14 + zarr v3 (previously crashed with
  ``FileExistsError`` on reopen).  The Optical Designer's save
  dialogs select between HDF5 and Zarr backends; before this
  patch, zarr writes could error mid-run on Windows.  No API change
  -- the GUI picks up the fix automatically.  See ``CHANGELOG.md``
  for details.

## [3.1.5] — 2026-04-20

### No GUI-side changes

- This release is a core-library bugfix for `.zmx` / `.txt` loaders
  (see `CHANGELOG.md` for details on the new
  `prescription['object_distance']` key).  The GUI does not surface
  this value directly, but prescriptions loaded via the Optical
  Designer's "Load Prescription" dialog now carry the correct
  obj-space distance in their returned dict -- downstream scripts
  and the GUI's own wave-optics preview stages benefit from the
  loader correction without any user-visible change.

## [3.1.4] — 2026-04-18

### Changed (wave-optics dock)

- **`Tilt-aware ray launch` checkbox default flipped from checked
  (True) to unchecked (False).**  Matches the core library's
  3.1.4 default flip of `apply_real_lens_traced(..., tilt_aware_rays=...)`.
  The previous default produced a reference-frame mismatch with
  the `preserve_input_phase=True` subtraction that output wrong
  fields on multi-mode inputs (post-DOE diffraction patterns,
  compound superpositions).  Existing GUI saves / sessions should
  re-run any wave-optics analysis with the new default to pick up
  the fix.  Advanced users doing rigorous off-axis characterisation
  of single-mode tilted inputs can still tick the checkbox.
- Checkbox tooltip updated to explain the new default + when to
  turn it on.

### Library side (transparent to GUI users)

See `CHANGELOG.md` for details.  Highlights: paraxial-magnification
Newton initial guess, experimental `inversion_method='backward_trace'`
opt-in for ~3x speedup on large grids, and the traced-lens
correctness fix for multi-mode inputs.

## [3.1.3] — 2026-04-17

Version-number unification release + two new controls in the
wave-optics dock exposing the core library's 3.1.3 additions.
Ships with **core library 3.1.3** (same version now -- see the
versioning note above) which is drop-in compatible with existing
GUI prescriptions.  See `CHANGELOG.md` for the full library entry.

### Added (wave-optics dock)

- **Precision selector** (Compute group): drop-down to choose between
  ``complex128`` (default, double precision) and ``complex64`` (single
  precision, half memory + ~2x FFT/phase-screen throughput).  The
  library applies its mod-2pi kernel-phase mitigation so complex64
  accuracy is bounded by FFT single-precision round-off (~-80 dB
  cumulative) rather than the phase-magnitude floor.  Tooltip lists
  the headroom tradeoff so users running deep-null / stray-light
  analysis below -60 dB know to stay at double.  The selected dtype
  propagates through the source-field allocation and all downstream
  library calls that preserve caller dtype (`apply_real_lens`,
  `apply_real_lens_traced`, `angular_spectrum_propagate`, `apply_mirror`).

- **Tilt-aware ray launch toggle** (Simulation Parameters group):
  checkbox exposing `apply_real_lens_traced(..., tilt_aware_rays=...)`.
  Defaults to True (matching the library default, and correct for the
  vast majority of inputs now that the smoothing fix makes multi-mode
  inputs robust).  Exposed primarily for A/B debugging and as an
  escape hatch if a pathological input slips past the smoothing.

### User-visible effects of the bundled library 3.1.3

In addition to the new UI controls above, the following improvements
apply automatically to every wave-optics run:

- **Wave-optics dock runs at large N (>= 16384) are faster and more
  memory-efficient.**  Each per-surface phase screen in `apply_real_lens`
  now uses a `numexpr`-fused multiply (optional dependency, falls back
  to numpy when absent) -- ~1.5-2x faster and ~3x lower peak memory at
  N=32768.  ASM propagation picks up a single-slot pyFFTW plan cache
  with in-place aligned buffers, eliminating the 30 s-TTL reallocation
  churn that previously fragmented Windows address space on multi-GB
  grids.
- **`apply_real_lens_traced` now converges correctly on multi-mode
  inputs.**  The 3.1.2 `tilt_aware_rays=True` default would silently
  zero-out the output field on post-DOE / interferometric inputs at
  large N (the per-pixel tilt extraction aliased at every fringe
  boundary, producing a chaotic entrance->exit spline that Newton
  couldn't invert).  The library now amplitude-weighted-Gaussian-smooths
  the tilt field so multi-mode inputs gracefully degenerate to the
  classical collimated launch while single-mode inputs (plane wave,
  Gaussian, MLA beamlets) keep their valid per-pixel tilts.  If you
  saw zero-output anomalies on a DOE-containing simulation, just re-run
  it in the GUI -- no prescription changes needed.
- **Optional complex64 mode** for ~2x memory / throughput at the cost
  of ~60 dB of effective cumulative dynamic range -- useful for
  design-verification sweeps at very large N where memory is the
  binding constraint.  Exposed as the Compute group's Precision
  selector in this release (see Added above).

### Fixed

- **Version line in the About dialog now reads correctly.** The GUI
  reads `__version__` from the package at runtime so it picks up the
  library bump (3.1.3) automatically -- no separate GUI-side version
  string to forget.

## [3.2.0] — 2026-04-16

Deep feature-gap pass: wired every high-leverage core-library
capability into the UI so the tool is usable for real design reviews
without dropping to Python.  Seven new analysis docks, four new
surface-form editors, a report generator, and an information-
architecture cleanup.

### Added — new docks

- **Through-focus dock** (`through_focus_dock.py`): axial Strehl /
  peak-intensity / RMS-radius / d4sigma plots with determinate
  progress, best-focus marker, CSV export.  Auto-populates its
  source field from the latest wave-optics run.
- **PSF / MTF dock** (`psf_mtf_dock.py`): log-scaled PSF + radial
  MTF plot, polychromatic-Strehl calculator across the Optimizer's
  wavelength list.  Pupil source is either the wave-optics
  exit-plane or a ray-trace-derived synthetic pupil.
- **Sensitivity dock** (`sensitivity_dock.py`): per-variable
  finite-difference d(merit)/d(var) with a horizontal bar chart
  sorted by |magnitude|.  Metric selectable (merit / RMS / EFL /
  BFL).
- **Interferometry dock** (`interferometry_dock.py`): Twyman-Green
  fringe simulator plus N-step phase-shift extraction with a
  measured-vs-truth residual plot.  Hardware / library sign
  conventions selectable.
- **Phase-retrieval dock** (`phase_retrieval_dock.py`): Gerchberg-
  Saxton + error-reduction runner with four target presets
  (Gaussian / top-hat / ring / Dammann grid), custom image loader,
  and convergence-history plot.
- **Field browser dock** (`field_browser_dock.py`): lists every
  saved plane in an HDF5/Zarr file, previews intensity + phase,
  and routes the selected plane into the Zernike, Interferometry,
  or PSF/MTF docks with one click.
- **Multi-Config dock** (`multiconfig_dock.py`): clones the current
  system into multiple configurations and drives
  `MultiPrescriptionParameterization` for joint optimisation
  (zoom steps, day/night, laser/imaging modes, ...).
- **Materials dock** (`materials_dock.py`): tabbed container that
  unifies the Glass Map (Abbe diagram) and User Library into a
  single entry point.  Original docks remain addressable from the
  View menu.

### Added — surface-form editors (`surface_editors.py`)

- **Asphere editor**: even-power coefficients up to r^20 with a
  live sag-profile preview.
- **Biconic editor**: Ry, ky overrides for anamorphic surfaces.
- **Freeform editor**: XY polynomial sag (i+j <= N) with grid UI.
- **Coating editor**: broadband/narrowband AR-coat model with
  wavelength range and target reflectance.

All four are reachable from the right-click context menu on the
surface sub-table.

### Added — optimizer upgrades

- Merit combos now include **ChromaticFocalShiftMerit**, **Match
  Ideal System** (via `MatchIdealSystemMerit.single_lens`),
  **Tolerance-aware** wrapper.
- **Wavelength / field weight editors** with photopic and cos^4
  presets.
- **Convergence plot** (merit vs iteration) drawn live inside the
  Optimizer dock; log-scale y-axis, auto-rescales.
- **Wave-merit gating**: selecting a wave merit and pressing
  "Local Optimize" redirects to the Wave Optimize path so the
  merit is actually honoured.

### Added — tolerance live histogram

- The Tolerance dock now redraws its RMS / EFL histograms every N
  trials (N = max(1, trials/40)) so the distribution forms live
  and the user can stop early once it looks converged.

### Added — snapshots compare

- "Compare selected to current" button: pops a side-by-side
  EFL / BFL / f-number delta table.

### Added — report export

- **Analysis -> Export design report (HTML)...**: one-page
  self-contained HTML with layout, spot diagram, ray-fan, Zernike
  plot, prescription table, and EFL / BFL / EPD / wavelength
  summary.  Images are embedded as base64 PNGs; file is shareable
  without separate assets.

### Added — preferences

- **Units menu** (SI vs Engineering).
- **Auto-retrace menu** (on / geometric-only / manual).
- **Error-routing policy dialog**: modal-on-error and
  status-bar-on-warn toggles go live on the diagnostics sink.

### Added — keyboard nudge

- **Shift+Up / Shift+Down** nudges the selected element's distance
  by +/-0.1 mm; **Ctrl+Shift+Up / Down** by +/-1 mm.  Works from
  anywhere in the window -- no need to click into the cell first.
  Undo-safe.

### Added — Thorlabs "find nearest part"

- **Insert -> Find nearest Thorlabs part**: ranks every catalog
  part by |dEFL| to the current system's paraxial EFL and shows
  the top 20.

### Added — empty-state CTAs (fix invisible-dependency traps)

- **Sliders dock**: when no variables are defined, shows a centred
  placeholder with a "Define variables..." button that opens the
  Optimizer's picker dialog directly.
- **Zernike dock**: adds a "From ray trace" button for fast
  geometric decomposition without requiring a prior wave-optics run.

### Changed

- **Analysis menu**: every analysis is now a "raise dock" shortcut
  rather than a dialog.  The old "Through-focus scan... future
  version" placeholder is gone.
- **Snapshots** now store the prescription alongside the state,
  enabling the new Compare workflow.
- **Wave-optics completion** auto-pushes the exit-pupil field into
  the Through-focus and PSF/MTF docks (was Zernike-only).

### Fixed

- **ProgressScaler signature**: the scaler now accepts both
  `(frac, msg)` and `(stage, frac, msg)` forms; previously being
  used as a `progress=` kwarg silently swallowed a TypeError, so
  the amp-phase progress inside `apply_real_lens_traced` was
  invisible in the UI.  Restored.

## [3.1.0] — 2026-04-16

Big usability pass driven by a UX deep-dive: undo/redo, snapshots,
diagnostics, autosave, prominent run forecast, optimizer checkbox grid,
Thorlabs catalog grouping, kerboard shortcuts, and many smaller
improvements.

### Added

#### Undo / Redo
- **Ctrl+Z / Ctrl+Y** at the window level; toolbar buttons too.
- Snapshot stack of depth 80 keeps the last ~80 mutations.  Loading a
  snapshot, importing a prescription, or running an optimizer all
  push to the stack so nothing is irrecoverable.

#### Snapshots dock
- Save the current design under a user-chosen name (Ctrl+B or the
  Snapshot button on the slider dock).
- Double-click a snapshot in the dock to restore it; restoring is
  itself undoable.
- A/B-comparison workflow without leaving the app.

#### Diagnostics dock + status-bar badge
- Replaces the scatter of silent ``except: pass`` blocks with a single
  log sink.
- Status-bar badge shows ``diag: ok`` (green) / ``diag: N new \u25CF`` (red)
  and clicking it raises the Diagnostics dock.
- The dock keeps a rolling 500-entry log with timestamps and tags.

#### Autosave + session restore
- Every system change writes ``~/.lumenairy/last_session.json``
  (debounced 1 s) so an accidental close doesn't lose the design.
- ``Edit \u2192 Restore Last Session`` brings it back; loading restores the
  full element list, source, wavelengths, EPD, and field angles.

#### Native JSON design format
- ``File \u2192 Save Design (JSON)`` (Ctrl+S) and ``File \u2192 Open Design (JSON)``.
- Self-contained, version-controlled, diffable — better than ``.zmx`` for
  shareable archived designs.
- ``File \u2192 Export Python Sim Script`` writes a runnable script via the
  new core ``codegen`` module.

#### Wave-optics dock overhaul
- **Lens-model selector**: choose between the inline ASM phase-screen
  pipeline (default, fastest), ``apply_real_lens`` (analytic, supports
  Fresnel + absorption), or ``apply_real_lens_traced`` (sub-nm OPD,
  ~10\u201330\u00d7 slower).  Routes the simulation through the chosen core
  function with full progress reporting.
- **Always-visible run forecast** strip above the Run button: lens
  model, grid, peak memory, estimated wall-clock time, disk size, with
  a colored ``[ok / HEADS-UP / CHECK BEFORE RUN]`` tag.
- **Recalibrated time/memory model** that finally agrees with measured
  runtimes for the v3.x ``apply_real_lens`` and ``apply_real_lens_traced``
  paths (calibration table in the new ``forecast_resources`` docstring).
- **Determinate progress bar** driven by the new core progress hooks:
  the bar smoothly advances through amplitude pass, ray-trace, Newton
  inversion, and field assembly.
- **Save planes: ON/OFF** segmented button next to Run \u2014 promoted
  from a buried checkbox so accidental large-N saves are harder to
  trigger.
- **Field-angle X/Y** now actually drive the source: a linear phase
  ramp ``exp(i (k_x X + k_y Y))`` is applied for every source type so
  off-axis simulations finally produce off-axis spots.

#### Optimizer dock
- **Checkbox grid dialog** replaces the two ``QInputDialog`` popups.
  One screen shows every (element, surface, parameter) with current
  values; tick what should be free, hit OK.  Bulk "Free all radii" /
  "Free all thicknesses" / "Clear all" shortcuts.
- **Wave-optimize free-variable mapping fix**: the old code mapped
  every UI ``distance`` variable to ``thicknesses[0]`` of the
  prescription regardless of which element it belonged to.  Now
  computes the correct flat thickness index per element.
- Variables and merit progress reported through the diagnostics sink
  on failure.

#### Slider dock
- **Per-slider \u00b1 range selector** (\u00b15 / 10 / 20 / 50 % for radii
  and thicknesses, \u00b10.2 / 0.5 / 1 / 2 for conics).  Pick the precision
  appropriate to each variable instead of a fixed \u00b150 %.
- **Live readout** now shows EFL, BFL, and f/# alongside the merit so
  the impact of a slider drag is visible without switching docks.
- **Snapshot button** in the toolbar saves the current parameter
  state under a name.
- Merit recomputation **debounced to 80 ms** so dragging large systems
  doesn't lock up the UI.
- ``opt_variables`` 3-tuple format is finally honoured (the old code
  unpacked them as 2-tuples and would crash when the optimizer dock
  produced ``distance`` variables).

#### Tolerance dock
- **Decenter tolerance is now actually applied** \u2014 the spinbox value
  was previously collected and discarded.  Lateral bundle offsets in
  X and Y are sampled at every trial.
- Per-trial failures route to the diagnostics sink so you can see why
  a Monte Carlo run lost trials.
- Anamorphic ``radius_y`` perturbed alongside ``radius`` (rather than
  silently snapping back to rotational symmetry).

#### Element table
- **Selection banner** above the surface detail panel makes it
  unambiguous which element\u2019s surfaces you\u2019re editing.
- **Throughput / stale indicator** in the info bar: ``[OK \u2713]`` /
  ``[STALE \u25CF]`` plus rays-alive / vignetting counts from the latest
  trace.
- **Wavelength and EPD** now use ``QDoubleSpinBox`` with debounced
  apply (no more "did my edit take?" wondering).
- **Coordinate-mode toggle** is now an unambiguous "Absolute
  coordinates [\u25A1]" checkbox-style button.
- **Group / Ungroup / Delete** buttons disable themselves when the
  current selection can\u2019t support the action.
- **Right-click on a surface** \u2192 *Propagate glass to all cemented
  faces* (handy for fixing imported doublets) or *Copy surface info to
  clipboard*.

#### Source panel
- Hides parameter rows that aren\u2019t used by the chosen source type
  (a plane wave no longer shows emitter-array fields).
- Field-angle X/Y inputs are always visible because they apply to all
  source types.
- Edits debounced via a 200 ms timer instead of firing on every
  keystroke.

#### Element insert dialogs
- Two-tier dialog with an **Advanced...** expander.  Quick path is
  unchanged (focal length + distance); advanced lets you override
  glass index, center thickness, and semi-diameter.
- **Repeat last insert** action (Ctrl+R) re-fires whichever insert
  dialog you used last.
- **Cylindrical lens** inserter now asks which axis carries the
  curvature (the old version hard-coded X).

#### Thorlabs catalog menu
- Grouped by family (LA, LB, AC, ACT, ...) with each entry labeled
  ``part   (f \u2248 NN mm)``.  Items within a family are sorted by
  focal length so you find lenses by ``f`` rather than by part number.

#### Layout views
- 2D and 3D layouts now call the core ``surface_sag_biconic`` instead
  of a hand-rolled paraxial sphere.  Conic, polynomial-aspheric, and
  biconic-Y contributions show up correctly and cannot drift from the
  ray tracer.
- 2D layout shows an empty-state hint card on first launch with
  "Insert \u2192 Lens \u2192 Plano-Convex Singlet" and shortcut tips.

#### Glass map dock
- Click-to-select a glass on the Abbe diagram, choose a target
  surface from a combo, click **Apply**.  No more guessing whether
  the click did anything.

#### Ray fan dock
- Field-curvature (the slow 21-field sweep) moved to an explicit
  **Compute** button so editing a system doesn\u2019t restart the sweep
  on every change.
- Standard matplotlib navigation toolbar (pan, zoom, save-PNG)
  added above the canvas.

#### Zernike dock
- New ``set_field`` entry point: pass a complex field and the dock
  runs ``wave_opd_2d`` with proper unwrap + reference-sphere
  subtraction before the decomposition.
- Defensive **wrap detector**: if the supplied OPD looks like raw
  wrapped phase (PV \u2264 2\u03c0) the dock prints a warning instead of
  fitting noise.
- Wave-optics dock auto-populates Zernike after a run.

#### Analysis menu
- Through-focus scan, chromatic focal shift, ray-traced Zernikes,
  spot-PNG export, Python-script export, and "Run Wave Optics now (F5)".

#### Toolbar
- Undo / Redo, Insert Lens, Insert Mirror, Run Wave Optics, Optimize
  added to the main toolbar in addition to New / Open / Retrace /
  Fit View.

#### Keyboard shortcuts
- Ctrl+Z / Ctrl+Y      undo / redo
- Ctrl+S               save as JSON
- Ctrl+L               insert plano-convex singlet
- Ctrl+Shift+L         insert achromatic doublet
- Ctrl+M               insert flat mirror
- Ctrl+R               repeat last insert
- Ctrl+D               delete element
- Ctrl+E               focus element table
- Alt+\u2191 / Alt+\u2193        move element up / down
- Ctrl+B               save snapshot
- Ctrl+T               retrace
- F5                   run wave optics
- ``Help \u2192 Keyboard Shortcuts`` lists them all.

### Changed

- All docks are now floatable: drag the title bar to pop a panel out
  into a separate window (useful on multi-monitor setups).
- **Cores stay byte-identical between the standalone library and the
  UI distribution.**  All UI changes live in ``lumenairy/ui/``;
  shared core changes (progress hooks, system element types, etc.) are
  applied to both copies.

### Fixed

- ``_populate_zernike_from_waveoptics`` no longer fails silently if the
  saved planes were renamed: falls back to the highest-z plane.
- Tolerance perturbations preserve ``radius_y``, ``conic_y``, and
  ``aspheric_coeffs`` instead of dropping them.
- Slider dock no longer misinterprets ``opt_variables`` 3-tuples
  (radius / thickness / distance / conic now all map correctly).

---

## [3.0.0] — 2026-04-16

Major release: wave-optics optimization, Zernike analysis panel,
anamorphic element support, expanded merit functions.

### Added

#### Zernike dock (`zernike_dock.py`, new)
- OSA-indexed Zernike decomposition of the exit wavefront.
- Interactive matplotlib bar chart of coefficients.
- Text summary with mode names and values.
- Configurable number of modes.

#### Wave-optics optimizer
- **Wave Optimize button** in the optimizer dock — runs
  `design_optimize` from the core library with wave-based merit terms.
- Wave merit types: Strehl ratio, RMS wavefront, chromatic focal shift,
  spot size.
- Global optimization methods: differential evolution, basin-hopping,
  dual-annealing (in addition to existing Nelder-Mead local).

#### Geometric merit types (expanded)
- RMS spot (existing).
- EFL target, BFL target — optimize toward a specific focal length.
- Seidel spherical — minimize third-order spherical aberration.
- Minimum thickness — constraint to prevent unphysical designs.
- Maximum f/# — constraint on system speed.

#### Anamorphic element support
- **Insert > Lens > Cylindrical Lens** — single-axis focusing.
- **Insert > Lens > Biconic Singlet** — independent x/y curvatures.
- **Radius Y** and **Conic Y** columns in the surface table for
  biconic surfaces.
- All existing analysis panels (spot diagram, ray fan, OPD, system
  data) handle biconic surfaces transparently.

#### Source configuration
- **Field angle X/Y** inputs on the source configuration panel for
  off-axis field point analysis.

### Changed

- **Core library updated to v3.0.0** — all new modules (vector
  diffraction, partial coherence, coatings, interferometry, freeform
  surfaces, ghost analysis, RCWA, multi-config, phase retrieval,
  through-focus/tolerancing, design optimizer) are available from the
  GUI's Python console and wave-optics dock.
- **`apply_real_lens_traced` exit-vertex fix** — the hybrid wave/ray
  lens model now focuses correctly for all lens geometries including
  cemented doublets and negative meniscus lenses.
- Removed organization name from application settings.

### Fixed

- Wave-optics dock correctly uses `apply_real_lens_traced` with the
  signed exit-vertex correction for all rear surface geometries.

---

## [2.5.0] — Prior release

- Initial GUI release.
- Element-based prescription editor with distance/tilt/decenter.
- 2D cross-section and 3D PyVista layout views.
- Spot diagram with Airy disc overlay.
- Ray fan / OPD analysis dock.
- System data (ABCD, EFL, BFL, f/#, NA).
- Glass map (interactive Abbe diagram).
- Local + global geometric optimizer.
- Live parameter sliders.
- Monte Carlo tolerance analysis.
- Wave-optics simulation panel (ASM, Fresnel, Fraunhofer, RS).
- HDF5 / Zarr output.
- Insert menu: plano-convex, biconvex, achromatic doublet, mirrors,
  DOEs, Thorlabs catalog lenses.
- Dark / Light / Midnight Blue themes.
- User library (materials, lenses, phase masks).

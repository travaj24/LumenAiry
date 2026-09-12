# UI audit — `lumenairy/ui/` (32 171 lines), `run_lumenairy_designer.py`, `requirements-gui.txt`

Environment note: **PySide6 is NOT installed** in this interpreter (nor PyQt5/6; `pyvistaqt` also
missing). Everything Qt-dependent was desk-checked. To get real numbers out of the physics glue I
wrote a 60-line `PySide6.QtCore` stub (`QObject` + a working descriptor-based `Signal`) under
`…/scratchpad/UI/stub/` and imported `lumenairy.ui.model` for real against numpy 2.4.6 / CPython
3.14. All measurements below marked "measured" come from that harness; scripts are in
`…/scratchpad/UI/t1…t11*.py`.

## Scope read

Line-by-line: `ui/model.py` (2929), `ui/waveoptics_dock.py` (2941), `ui/analysis.py` (395),
`ui/psf_mtf_dock.py` (physics half), `ui/optimizer_dock.py` (worker + merit plumbing),
`ui/element_table.py` (source form + info bar), `ui/layout_2d.py` (source glyphs + `_draw_rays`),
`ui/diagnostics.py`, `run_lumenairy_designer.py`, `requirements-gui.txt`.
Skim + targeted reads: `main_window.py` (3625), `layout_3d.py`, `spot_field_dock.py`,
`coronagraph_dock.py`, `log_viewer_dock.py`, `lg_aberration_dock.py`, `footprint_dock.py`,
`workspace.py`, and structural/grep sweeps over all 49 modules (QThread usage, `finished` signal
shadowing, `isInterruptionRequested`, global-state setters, matplotlib/pyplot, QTimer, sag/ABCD
duplication, `wavelength_nm` flow, `to_prescription()` consumers).
Cross-checked against `CONVENTIONS.md` §7, `raytrace/world.py`, `raytrace/trace.py`,
`raytrace/seidel.py`, `io/prescriptions_builders.py`, `io/prescriptions_zemax.py`,
`propagators/fft_infra.py`, `elements/lenses.py`.

---

## Findings

### **[P0] `SystemModel.to_prescription()` silently strips `is_mirror` and leaks Zemax-signed negative thicknesses into the legacy `surfaces`/`thicknesses` keys** — `lumenairy/ui/model.py:2761-2782`

The legacy export loop emits only
`radius, conic, aspheric_coeffs, glass_before, glass_after, radius_y, conic_y, aspheric_coeffs_y`.
`build_trace_surfaces()` marks mirrors with `is_mirror=True` and gives post-mirror surfaces
**negative** (Zemax-signed) thicknesses — both are dropped. A mirror therefore round-trips as an
air→air refracting surface, i.e. a no-op, and the fold's negative gap becomes a literal
backwards propagation.

Evidence (measured, `t2_mirror.py`, concave mirror R=−200 mm + N-BK7 singlet):

```
legacy surfaces:  {'radius': -0.2, 'glass_before': 'air', 'glass_after': 'air'}   is_mirror key present: False
thicknesses:      [-0.03, 0.004]
ABCD from EXPORTED prescription: efl= 0.15886  bfl= 0.15620
ABCD from UI internal surfaces : efl= 0.17877  bfl= 0.12780      <-- 11 % / 22 % apart
```

And with a 45° fold (`t11.py`) the whole fold geometry evaporates:

```
local trace surfaces: cb t=0.0 | mirror t=0.0 | cb t=0.04 | S1 t=0.003 | S2 t=0.0
legacy export       : 3 surfaces, thicknesses [0.0, 0.003]
LOST (the cb_post air gap): 0.04 m
```

Impact: this is the single prescription dict handed to **18 call sites across 14 docks**
(`analysis.py:332`, `caustic_dock:230`, `coherence_dock:187,315`, `ghost_dock:269,553`,
`jones_pupil_dock:305`, `lg_aberration_dock:114`, `library_dock:183` (**saved to the user's lens
library**), `multiconfig_dock:185`, `optimizer_dock:916`, `psf_mtf_dock:209,354`,
`snapshots_dock:102`, `waveoptics_dock:442,2099`, `main_window:1612,2069,2119,2172,2735,3137`,
and `model.recommend_grid`). `surfaces_from_prescription` consumes `surfaces` + `thicknesses`
verbatim, so every one of those docks analyses a *different optical system* than the one the
layout and spot diagram show. The `elements` key (added 3.7.0) does carry `element_type:'mirror'`
correctly — only the legacy pair is broken, which is why `export_zemax_zmx` survives and
everything else does not.

Fix: emit `is_mirror`, `semi_diameter`, and `is_stop` on each `rx_surfaces` entry, and either
export physical-positive thicknesses or refuse to emit the legacy keys for a system containing a
mirror (`allow_unfolded_equivalent` already exists as the library's opt-in flag for this exact
ambiguity — see `apply_real_lens`'s error text). Also delete the dead `elif` at
`model.py:2779-2782`: both branches are `thicknesses.append(s.thickness)`, so the comment's claim
that the cb's thickness "is absorbed into the next refractive surface's gap" is false — cbs are
`continue`d at line 2764 and their thickness is simply lost.

---

### **[P0] Per-surface semi-diameters are re-indexed onto the wrong surfaces whenever the system contains a mirror** — `lumenairy/ui/model.py:2761-2772` + `raytrace/trace.py:527-533`

`to_prescription()` omits `semi_diameter` from the legacy `surfaces` dicts, so
`surfaces_from_prescription` falls back to its `elements`-list matcher:

```python
refr_elems = [e for e in elements if e.get('element_type') == 'surface']
if i < len(refr_elems):
    elem_sd = refr_elems[i].get('semi_diameter', np.inf)
```

`refr_elems` **excludes** mirrors, but `i` runs over the full `surfaces` list which **includes**
them. Every mirror shifts the mapping by one.

Evidence (measured, same `t2_mirror.py` system — mirror sd 25 mm, lens sd 6 mm):

```
surfaces_from_prescription ->
   R=-0.2 (the mirror)  sd=0.006      <-- got the LENS's semi-diameter
   R=0.08 (lens S1)     sd=0.006
   R=inf  (lens S2)     sd=0.0127     <-- fell through to aperture_diameter/2
```

Impact: vignetting is computed against the wrong apertures in every prescription-consuming dock;
a 50 mm mirror is clipped to 12 mm. Fix: emit `'semi_diameter': sd_m` in each `rx_surfaces` entry
(then the per-surface key wins at `trace.py:525` and the fragile `elements` matcher is bypassed).

---

### **[P1] Point-source ray bundle is degenerate: all rays collapse onto the 45° diagonal (the azimuth loop variable is never used)** — `lumenairy/ui/model.py:2357-2371`

```python
for t in theta:                                   # t is NEVER used
    all_x.append(0.0); all_y.append(0.0)
    all_L.append(frac * semi_ap / obj_dist + tilt_L)
    all_M.append(frac * semi_ap / obj_dist + tilt_M)
```

The intent was obviously `L = ρ·cos(t)`, `M = ρ·sin(t)`.

Evidence (measured, `t4_pointsrc.py`, `num_rings=3, rays_per_ring=8`):

```
n rays: 25     unique (L,M) pairs: 4     L==M everywhere: True
L = [0.021167 ×8, 0.042333 ×8, 0.0635 ×8, 0.]
```

Impact: with `source_type='point_source'` selected (one of seven first-class UI source types; the
Insert ▸ Source preset defaults it to 1000 mm) the spot diagram, `spot_rms`, `spot_geo_radius`,
the ray-fan and the 2-D ray overlay are all computed from `num_rings` distinct pencil directions
lying on the x=y line, each duplicated `rays_per_ring` times. The marginal ray is also √2 too
steep (both L and M carry the full `semi_ap/obj_dist`), so the pupil is over-filled by 41 %.
Secondary: the divergence is derived from `SourceDefinition.object_distance_mm` while the rays are
launched at world z=0 and the first surface sits at `elements[1].distance_mm` — two independent
knobs for the same quantity, so the pupil fill is only correct when the user happens to keep them
equal. Fix: `all_L.append(frac*semi_ap/obj_dist*np.cos(t) + tilt_L)` (and `sin` for M), and derive
`obj_dist` from the element geometry instead of a second field.

---

### **[P1] A wave-optics run with the default backend permanently disables pyFFTW *and* SciPy FFT for the whole process** — `lumenairy/ui/waveoptics_dock.py:566-573`

```python
backend = cfg.get('backend', 'numpy')
from ..propagators import fft_infra as _fft_infra
_fft_infra.USE_PYFFTW = False
_fft_infra.USE_SCIPY_FFT = False
if backend == 'pyfftw': _fft_infra.USE_PYFFTW = True
elif backend == 'scipy': _fft_infra.USE_SCIPY_FFT = True
```

Library defaults are `USE_PYFFTW = PYFFTW_AVAILABLE` (True — pyfftw 0.15.1 is installed) and
`USE_SCIPY_FFT = True` (`propagators/fft_infra.py:150,155`). The dock's combo is built as
`backends = ['NumPy FFT']` first (`waveoptics_dock.py:1637`), so **index 0 / the default** maps to
`backend='numpy'` and clears both flags. The write is done from the worker thread, is global to
the module, and is never restored. `set_max_ram(mem_limit)` at line 558-559 is the same pattern.

Impact: one default Run silently downgrades every subsequent FFT in the session — other docks, the
optimizer's wave leg, the REPL dock, and anything the user scripts — from multi-threaded
SciPy/pyFFTW to single-threaded numpy, with no UI indication. Fix: save/restore around the run
(`try/finally`), or better, route through the library's own `set_fft_fallback` scoped-context API
and leave "library default" as combo index 0.

---

### **[P1] The lens-model router falls back to the crude per-surface ASM loop on *any* exception and never tells the user** — `lumenairy/ui/waveoptics_dock.py:716-727`, reported at `:2916`

```python
except Exception as e:
    try:
        from .diagnostics import diag
        diag.report('waveoptics-lens-router', e, context=f'lens_model={lens_model}')
    except Exception: pass
    used_lens_router = False        # <-- falls through to the inline loop
```

`_on_finished` prints `Method: {self.combo_method.currentText()}` and `Backend: …` but **never
reports which lens model actually ran**.

Evidence (measured, `t8_docks.py`): a folded system makes `apply_real_lens` raise by design —

```
apply_real_lens raises: ValueError: apply_real_lens: prescription has 1 mirror element(s) but
apply_real_lens only walks refracting surfaces. … set prescription['allow_unfolded_equivalent']=True …
```

so every folded design silently downgrades from the requested analytic/traced/Maslov model to the
thin-screen ASM loop (which then unfolds the mirrors itself via `_filter_wave_optics_surfaces`).
The user sees a plausible PSF labelled with their chosen method. Fix: surface the fallback in the
summary panel and in `results['lens_model_used']`; better still, honour the library's explicit
`allow_unfolded_equivalent` handshake rather than catching its refusal.

---

### **[P1] `_on_finished` slices the focal field with the *input* grid N — crashes or mis-plots for MFT and detector runs** — `lumenairy/ui/waveoptics_dock.py:2858-2894`

```python
N = results['N']          # the INPUT grid size
c = N // 2 ; w = N // 8
crop   = I_log[c-w:c+w, c-w:c+w]
x_um   = (np.arange(N) - N/2) * dx * 1e6
I_slice = I_focus[c, :]                     # IndexError when I_focus is smaller than N
```

But `I_focus` is `mft_N_out × mft_N_out` on the three `*-mft` methods
(`waveoptics_dock.py:580, 1093`) and is re-binned to the detector pixel pitch when
`detector_apply` is on (`:1043-1048`, `apply_detector` returns an image at `pixel_pitch`).
`results['N']` is never updated for either. Fix: `N = I_focus.shape[0]` in `_on_finished` (and
store the true output N in `results`).

---

### **[P1] `Insert ▸ Source ▸ …` (six menu actions) raises `AttributeError` — `SystemModel.source` is a read-only property** — `lumenairy/ui/main_window.py:3227` (menu wiring `:1184-1195`)

```python
self.model.source = SourceDefinition(kind, wavelength_nm=self.model.wavelength_nm, **kwargs)
```

`SystemModel.source` is declared `@property` at `model.py:544-548` with no setter.

Evidence (measured, `t6_srcprop.py`):
`assignment FAILS: AttributeError property 'source' of 'SystemModel' object has no setter`

There is no `try` around it, so all six Insert ▸ Source presets are dead and throw out of the Qt
slot. Fix: call `self.model.set_source(...)` (which exists, checkpoints, invalidates and emits).

---

### **[P1] Every source-parameter edit resets the source wavelength to 1310 nm and discards the polarization** — `lumenairy/ui/element_table.py:675-688`

`_apply_source_params` rebuilds the `SourceDefinition` from scratch out of the 13 `_SRC_PARAMS`
line edits (`element_table.py:415-456`). `wavelength_nm` and `polarization` are **not** among
them, so `SourceDefinition.__init__`'s defaults (`1310.0`, `None`) win.

Compounding this, `SystemModel.set_wavelength()` (`model.py:813-820`) never syncs
`source.wavelength_nm` either — no code path anywhere in `ui/` does.

Evidence (measured, `t5_wv.py` / `t7_emitter.py`):

```
after set_wavelength(632.8): model = 632.8   source = 1310.0
Source built at wavelength: 1.31e-06   model.wavelength_m: 6.328e-07   -> MISMATCH
max |phase diff| between stale-wv and correct-wv point source: 0.216 rad
wavelength_nm after edit-rebuild: 1310.0 ; polarization after edit-rebuild: None
```

Impact: the wave-optics worker builds `E` via `src.to_source(...)` at the **source's** wavelength
(`model.py:157`) and then propagates at the **model's** wavelength (`waveoptics_dock.py:527`). For
`point_source` / `fiber_mode` / tilted sources the launched field carries the wrong spherical
phase and the wrong carrier tilt. Fix: make `wavelength_nm` a derived read-through to the model
(or sync it in `set_wavelength` and preserve it in `_apply_source_params`).

---

### **[P1] `emitter_array` source is unusable: the element table casts `emitter_nx/ny` to float, `to_source` then raises and is swallowed into a plane wave** — `lumenairy/ui/element_table.py:684` + `model.py:212-213`

`_apply_source_params` does `kwargs[key] = float(val)` for *every* field, including the two
integer counts. `to_source` then does `for iy in range(self.emitter_ny)`.

Evidence (measured, `t7_emitter.py`):
`emitter_nx type: float 12.0` → `to_source RAISES: TypeError 'float' object cannot be interpreted
as an integer`.

The wave-optics worker catches this in the broad `except Exception` at
`waveoptics_dock.py:628-638` and silently substitutes an EPD-clipped **plane wave**, so the user
gets a plane-wave PSF labelled "Emitter array 12×12".
Related crash: `layout_2d.py:607-608, 703-704` does `nx = max(1, min(7, src.emitter_nx))` then
`range(nx)` — with a float count ≤ 7 that is a `TypeError` inside the scene rebuild.
Fix: `int(...)` for the count fields in `_apply_source_params` (or a per-field caster table), and
`int()` guards in `layout_2d`.

---

### **[P1] `_build_trace_surfaces_world()` drops every inter-element air gap, so the paraxial-focus fallback and any ABCD taken on the world list are wrong** — `lumenairy/ui/model.py:2244-2248, 2318-2417, 2001-2019`

```python
if si < len(elem.surfaces) - 1:
    thick_m = srow.thickness * 1e-3
else:
    thick_m = 0.0            # air gap to the NEXT element is never carried
```

`trace_world` itself is fine (it uses `world_origin`, which *does* include the gaps — verified),
but `run_trace` (`model.py:2400`) and `build_run_trace_world_surfaces` (`model.py:2003`) both call
`find_paraxial_focus(<world list>, wv)` when the Detector distance is 0.

Evidence (measured, `t10_abcd_cb.py`, two singlets 40 mm apart):

```
world thicknesses (m): [0.003, 0.0,  0.003, 0.0]
local thicknesses (m): [0.003, 0.04, 0.003, 0.0]
find_paraxial_focus(world) = 0.058344      find_paraxial_focus(local) = 0.040112   (+45 %)
system_abcd(world) efl = 0.061577          system_abcd(local) efl = 0.072971       (-16 %)
```

Impact: with the detector at 0 the image plane lands 18 mm past focus and every spot/RMS/Airy
number in the summary is taken there. Six docks consume this list
(`footprint_dock:98`, `rayfan_dock:106`, `spot_field_dock:150`, `tolerance_dock:69`,
`distortion_dock:129`, `wavefront_map_dock:104`). Fix: carry the gap to the next element on the
last surface of each element, exactly as `_build_trace_surfaces_internal` already does at
`model.py:2160-2172`.

---

### **[P1] "Pupil from ray trace" in the PSF/MTF dock is doubly broken: wrong attribute name, and image-plane coordinates used as pupil coordinates** — `lumenairy/ui/psf_mtf_dock.py:204-286`

(a) Line 221 reads `rays.opl`; `RayBundle` has no such attribute — it is `opd`
(`raytrace/trace.py:802-808`).
(b) Lines 217-237 bin `result.image_rays.x/y` (positions at the **image plane**) into a grid of
diameter `self.sm.epd_m` — pupil coordinates.

Evidence (measured, `t9_psfpupil.py`, 50 mm BK7 singlet, 288 rays):

```
image_rays attrs: ['L','M','N','alive','copy','error_code','n_rays','opd', …]   has .opl: False
image-plane ray radius: max|r| = 2.50e-04 m   (EPD/2 = 1.27e-02 m)
distinct pupil pixels filled: 31 of 65536
```

The exception is caught by the caller (`psf_mtf_dock.py:172-179`) and shown as "Pupil load
failed: AttributeError…", so today the feature is simply dead. Fixing only the attribute name
would produce a 31-pixel blob at the centre of a 256×256 "pupil" and a meaningless PSF/MTF. Fix:
use `rays.opd`, and take the ray heights at the **exit pupil / last surface**
(`result.ray_history[-2]`) rather than at the image plane. Line 209-211's `to_prescription()` →
`surfaces_from_prescription` → `system_abcd` also inherits the P0 above for folded systems.

---

### **[P1] The optimizer worker mutates the live shared model on every scipy probe — the documented "S4-7 fix" only covers the final write-back** — `lumenairy/ui/model.py:2483-2484, 2442-2450`; `ui/optimizer_dock.py:59-141`

`OptimizeWorker.run()` (background thread) → `model.run_optimization(..., apply_result=False)` →
`scipy.minimize(self.merit_function, …)` → `merit_function` → `set_variable_values(values)` which
writes `elem.distance_mm` / `surfaces[i].<field>` **and** calls `self._invalidate()`, which calls
`recompute_element_frames()` and rewrites `elem.origin` / `elem.R` on every element — hundreds of
times a second, from the worker thread, while the GUI thread paints from exactly those attributes
(`layout_2d._draw_rays` reads `surface_frames_2d_mm()`, `layout_3d` reads
`element_frames_3d_mm()`, `element_table` reads `display_value_absolute`). `_invalidate` also
nulls `_flat_surfaces_cache` under any dock that is mid-iteration over it.

The class comment at `optimizer_dock.py:85-88` — "it never mutates the shared live model off the
GUI thread" — is measurably false; `apply_result=False` only changes what happens *after*
`minimize` returns. Fix: deep-copy the model (or just the element list) into the worker, optimize
the copy, and hand `result.x` back — the pattern `WaveOpticsWorker._snapshot_model` already
establishes for the wave dock.

---

### **[P1] 14 of 16 worker threads ignore `requestInterruption()`, so quitting mid-run hits `QThread: Destroyed while thread is still running`** — `lumenairy/ui/main_window.py:40-71`

`_shutdown_dock_workers` (called from `MainWindow.closeEvent`) does
`th.requestInterruption()` then `th.wait(2000)` on every dock's `_worker`. Only
`waveoptics_dock.py:483` and `through_focus_dock.py:67` actually poll
`isInterruptionRequested()` (grep over `ui/`). `OptimizeWorker` polls its own unrelated
`CancellableProgress.should_stop` flag (`optimizer_dock.py:122`), which `requestInterruption()`
does not set. Everything else (`ToleranceWorker`, `GlobalSearchWorker`, `WaveOptimizeWorker`,
`_AOClosedLoopWorker`, `_PhaseRetrievalWorker`, `_CoronagraphWorker`, `_KoehlerWorker`,
`_CoherenceAnalysisWorker`, `_GhostWorker`, `CausticWorker`, `_MultiConfigWorker`,
`RichardsWolfWorker`, `_PolyStrehlWorker`, `_OPDWorker`) never checks anything.

Impact: the 2 s wait times out and Qt aborts the process during shutdown; a tolerance Monte-Carlo
or AO closed-loop run (minutes) reliably reproduces it, and any in-flight HDF5/Zarr write is
truncated. Fix: add an `isInterruptionRequested()` poll at each worker's natural loop boundary
(most already have one), and make `CancellableProgress` read from it.

---

### **[P2] Three QThread subclasses still shadow the built-in `QThread.finished` signal** — `ui/optimizer_dock.py:69`, `ui/optimizer_dock.py:1468`, `ui/tolerance_dock.py:41`

`waveoptics_dock.py:390-394` documents this exact defect (v5.17 audit P3-62) and renames its own
signal to `finished_result` — the three siblings were never migrated. `worker.finished` now binds
to the custom `Signal(bool, str)` / `Signal(object)`, so the canonical
`worker.finished.connect(worker.deleteLater)` idiom silently attaches to the wrong signal and
never fires when `run()` raises. Fix: rename to `finished_result` in all three.

---

### **[P2] "Start at / End at" element-range controls in the wave-optics dock are dead** — `ui/waveoptics_dock.py:552-553` vs `:2591-2604, 1620-1628`

```python
start_idx = cfg.get('start_elem', 0)
end_idx   = cfg.get('end_elem', self._snap['n_elements'] - 1)
```

Both locals are assigned and then **never referenced again** in the 685-line `_run_impl` (grep for
`start_idx|end_idx` in the file returns only these two lines plus the dock-side producer). The
user restricts the propagation to a sub-range and gets the full system. Fix: slice `trace_surfs`
by the range (and clamp `bfl` accordingly), or remove the controls.

---

### **[P2] `power_in` is measured at the wrong plane when the user unchecks "Source" in the save-plane list** — `ui/waveoptics_dock.py:1062`

```python
power_in = beam_power(planes[0]['field'] if planes else E, dx)
```

`planes[0]` is only the source plane when `save_plane_flags.get('Source', True)` was true
(`maybe_save`, `:666-670`). Uncheck it and `planes[0]` becomes `LensExit` or the first saved
surface, so the "Throughput: xx %" line at `:2915` silently becomes ~100 %. Fix: capture
`power_in` from `E` immediately after construction, independent of the save flags.

---

### **[P2] `build_run_trace_world_surfaces(image_distance=…)` ignores its own argument whenever a Detector with non-zero distance exists** — `ui/model.py:2005-2021`

The Detector branch is tested first and `image_distance` is only consulted in the `elif`. Callers
that explicitly ask for the paraxial focus get the detector plane instead:
`spot_field_dock.py:150` passes `image_distance=bfl` (computed correctly from the **local** list at
`:136-138`) and then draws an Airy-radius overlay at `:187` for the focal plane — on a spot diagram
rendered at the detector. For a default new system that is 100 mm vs a 97.3 mm BFL (measured,
`t1_singlet.py`). `distortion_dock.py:129` has the same shape. Fix: let an explicit
`image_distance` override the detector, or rename the parameter to `fallback_image_distance` and
have the docks stop passing it.

---

### **[P2] `get_variable_values()` / `set_variable_values()` disagree on indexing when an optimization variable goes stale** — `ui/model.py:2431-2450`

`get_variable_values` **skips** entries whose `elem_idx`/`surf_idx` is out of range (returning a
shorter vector); `set_variable_values` indexes `values[i]` with `i` running over the **unfiltered**
`opt_variables`. Delete an element that had a variable attached (`delete_element` does not prune
`opt_variables`) and the next optimize either assigns a value to the wrong parameter or raises
`IndexError` — swallowed at `model.py:2654` into a generic failure message. Fix: prune
`opt_variables` in `delete_element`/`move_element`, and make both methods iterate the same
filtered list.

---

### **[P2] `_update_forecast` runs a blocking ASM benchmark on the GUI thread, wired to 14 widget signals** — `ui/waveoptics_dock.py:2480` → `:48-88`

`_local_asm_baseline_ms()` runs one warm-up plus two timed `angular_spectrum_propagate` calls at
N=512 (three 512² complex FFT pairs) the first time any of the 14 connected widgets changes
(`spin_N`, `spin_dx`, `combo_method`, `chk_mft`, `chk_bandlimit`, `chk_unfold_mirrors`,
`chk_ignore_lateral_cbs`, `spin_raysub`, `combo_mem`, `combo_precision`, and 3 plane checkboxes —
lines 1276-1955). Result is cached in the module global `_CALIBRATED_ASM_MS_AT_1024`, so it is a
one-time freeze of a few hundred ms — but it happens on the user's first keystroke in the dock,
and the "Recalibrate" button (`_recalibrate`, `:1867`) re-arms it synchronously. Fix: run the
calibration once in a `QThread` at dock construction, or seed with the 12 ms fallback and refine
in the background.

---

### **[P2] `File ▸ New` calls `SystemModel.__init__()` on the live QObject, silently wiping display preferences** — `ui/main_window.py:2184-2193`

```python
self.model.__init__()
```

Re-running `__init__` resets `prefs` (`bg_2d`, `bg_3d`, `ray_color`, `ray_use_wavelength`,
`accent`, `theme`), `lens_options`, `auto_retrace_mode`, `unit_preference`, `snapshots`,
`opt_variables`, `wavelength_weights`, `field_weights` — none of which the user expects "New
system" to touch — and re-invokes `QObject.__init__(parent)` on an already-constructed C++ object.
Fix: add a `SystemModel.reset_design()` that clears only the element list / wavelength / EPD /
history.

---

### **[P2] Duplicate air-gap semantics between `set_display_distance` and `set_element_absolute_field`** — `ui/model.py:789-810` vs `:755-765`

`_prev_element_back_vertex_world` was introduced (v4.15 P1-UI-4) as the single source of truth for
"previous element's back vertex", and `set_element_absolute_field:781` uses it — but
`set_display_distance:806` re-derives the same quantity inline as
`prev_z + prev_elem.internal_thickness_mm`, i.e. it re-implements the helper rather than calling
it. They agree today (both use the `surfaces[:-1]` slice) but this is exactly the drift the helper
was created to prevent. Fix: call the helper and project `[2]`.

---

### **[P3] `_apply_real_lens_asm_equiv` does not reproduce the calibration points in its own docstring** — `ui/waveoptics_dock.py:272-281`

```python
"""Empirically: 2 surfaces -> 1.1 ASM, 3 surfaces -> 2.2 ASM."""
return max(n - 1, 1) * 1.0 + 0.2 * n
```

n=2 → 1.4 (claimed 1.1, +27 %); n=3 → 2.6 (claimed 2.2, +18 %). Answering the audit brief's
question directly: this is **not** a physics quantity or an ASM step count used by the
propagation — it is a pure cost-model coefficient feeding the Time estimate in the forecast panel
and nothing else (`forecast_resources` only; grep confirms two call sites, both in the time
model). So the error is cosmetic: forecasts for the `real_lens` / `real_lens_traced` paths read
~20 % pessimistic. Fix: `return max(n - 1, 1) * 0.9 + 0.1 * n` fits both stated points, or update
the docstring to the implemented model.

### **[P3] `analysis.py`'s "Image-plane WFE" block is unreachable** — `ui/analysis.py:334-335`

```python
if presc and presc.get('surfaces') and presc.get('object_distance', 0) > 0:
```

`to_prescription()` never emits an `object_distance` key (the returned dict is
`name, aperture_diameter, surfaces, thicknesses, elements, all_thicknesses, coord_breaks` —
`model.py:2893-2905`), so the condition is always False and the whole
`eval_image_plane_wfe` / `remove_low_order_aberrations` panel (lines 336-354) is dead.

### **[P3] `layout_2d` emitter-array glyph multiplies the column offset by zero** — `ui/layout_2d.py:614`

`cx = x0 + (ix - (nx - 1) / 2) * pitch_px * 0.0` — the `ix` loop draws `nx` identical overlapping
dots. Presumably deliberate (the side view projects x away) but then the loop should be removed.

### **[P3] `_suppress_history` is write-only** — `ui/model.py:518-521, 1420-1424`

Set to `False` in `__init__`, read in `_checkpoint`, never set to `True` anywhere. Its docstring
names `group/ungroup, load_prescription` as the users; all three just call `_checkpoint()` once
directly. Dead flag.

### **[P3] `run_optimization`'s completion message always labels the merit in µm** — `ui/model.py:2650`

`f'Merit: {result.fun*1e6:.3f} um'` — correct for `rms_spot`, meaningless for `efl_target`,
`bfl_target`, `seidel_spherical`, `min_thickness`, `max_fnumber` (dimensionless squared errors).

### **[P3] `min_thickness` merit carries a constant offset** — `ui/model.py:2527-2533`

Iterates all `elem.surfaces` including each element's trailing surface, whose `thickness` is 0 by
the model's own convention (the air gap lives on `Element.distance_mm`), so every element
contributes a fixed `(1.0 - 0)**2 = 1`. Harmless for a gradient-free search but the merit can
never reach 0 and its printed value is not the physical min-thickness violation.

### **[P3] Worker snapshot aliases the model's cached `Surface` objects** — `ui/waveoptics_dock.py:434` and `:307-380`

`snap['trace_surfs'] = list(model.build_trace_surfaces() or [])` copies the *list*, not the
`Surface` objects, and `_filter_wave_optics_surfaces` only copies entries it has to flip the sign
on. The class comment claims "no reference into the model survives into run()". `_run_impl` only
reads them today, so nothing is broken — but the invariant the comment asserts does not hold, and
`_filter_wave_optics_surfaces`'s "callers can safely mutate" docstring is wrong for the elements.

### **[P3] `diagnostics.report` hard-codes a Windows path separator** — `ui/diagnostics.py:78`

`last.filename.split(chr(92))[-1]` — on POSIX the full path is printed. Use `os.path.basename`.

---

## Performance opportunities

1. **Import cost.** `import lumenairy` is 2.86 s (measured) and does **not** pull `lumenairy.ui`
   (verified: `ui in sys.modules: False`) — good. But `main_window.py:20-37` eagerly imports 14
   dock modules, and **30 of the 49 UI modules import matplotlib at module scope**;
   `waveoptics_dock.py:26-27` additionally pulls `matplotlib.backends.backend_qtagg`. `python -X
   importtime -c "import matplotlib.figure"` reports 17.5 s cumulative cold on this box (sub-second
   warm). Moving the matplotlib imports and the non-default docks behind a lazy
   `_create_docks`-time import would cut startup materially.
2. **`_local_asm_baseline_ms` on the GUI thread** — see P2 above.
3. **`_on_finished` plots without downsampling** — `I_log[c-w:c+w, c-w:c+w]` with `w = N//8` is an
   N/4 × N/4 imshow: 1024² for N=4096, 2048² for N=8192. Bounded but large; a `max(1, w//256)`
   stride would cost nothing visually.
4. **`autosave_session` deep-copies + JSON-writes on the GUI thread** every 1 s after the last
   change (`main_window.py:189-193` → `model.py:1514-1532`). Fine for small designs; consider a
   thread or a dirty-check.
5. **`merit_function` rebuilds the entire `Surface` list twice per probe**
   (`model.py:2485-2497` then again per wavelength × field at `:2555-2562`) and re-runs
   `find_paraxial_focus` per probe. For a 3-wavelength × 5-field merit that is 15 rebuilds of the
   full surface list per scipy evaluation.
6. **`to_prescription()` is called per-widget-event by several docks** (e.g. `_update_forecast`
   → `build_trace_surfaces` is cached, but `recommend_grid` → `to_prescription` is not). Cache it
   on the same invalidation key as `_flat_surfaces_cache`.

---

## Code organization observations

### Module map (`wc -l lumenairy/ui/*.py`, 32 171 total)

| Lines | Module | Role |
|------:|--------|------|
| 3625 | `main_window.py` | shell: menus (344 ln), `_create_docks` (375 ln), file IO, themes, 40+ `_ins_*` |
| 2941 | `waveoptics_dock.py` | forecast model + `WaveOpticsWorker` (`_run_impl` = **685 ln**) + dock (`__init__` = **658 ln**) |
| 2929 | `model.py` | `SurfaceRow`/`SourceDefinition`/`Element`/`SystemModel` — the only non-view module |
| 1909 | `optimizer_dock.py` | 3 QThread workers + variable grid + wave-leg |
| 1325 | `element_table.py` | 2 QAbstractTableModels + surface sub-editor + source form |
| 1118 / 995 | `layout_2d.py` / `layout_3d.py` | views (both correctly delegate sag to `elements.lenses.surface_sag_biconic`) |
| 959…823 | `algebra_dock`, `ao_dock`, `phase_retrieval_dock`, `coherence_dock`, `coronagraph_dock` | heavy analysis docks |
| 782…396 | 14 mid-size docks (`coatings`, `ghost`, `wavefront_map`, `tolerance`, `chebyshev_fit`, `lens_options_dialog`, `through_focus`, `slider`, `zernike`, `log_viewer`, `psf_mtf`) | |
| 395…162 | `analysis.py`, `surface_editors`, 20 small docks | |
| 647 / 222 / 63 / 22 | `workspace.py`, `diagnostics.py`, `materials_dock.py`, `__init__.py` | |

### Observations

* **Model/view separation is good at the top** — `model.py` imports only `PySide6.QtCore`
  (`QObject`, `Signal`), numpy, and library code; I was able to import and exercise it with a
  60-line Qt stub. That is the right shape and should be defended.
* **…but the model has grown view and controller responsibilities**: `geo_merit_type` /
  `geo_merit_target` are class attributes on `SystemModel` set by the optimizer dock
  (`model.py:2480-2481`); `prefs` holds 2-D/3-D background colours and the theme name; `to_source`
  duplicates source-construction logic that `waveoptics_dock` then re-implements as a fallback
  (`:628-638`). Proposed split: `ui/model/system.py` (elements + frames + caches),
  `ui/model/prescription_io.py` (`load_prescription` / `to_prescription` — 430 lines of pure
  translation with no Qt), `ui/model/optimize.py` (merit + `run_optimization`), leaving `prefs` in
  a separate `ui/preferences.py`. `prescription_io` becomes unit-testable without Qt at all.
* **Three giant functions dominate the diff surface**: `WaveOpticsWorker._run_impl` (685 ln,
  10 nesting levels, five different propagator dispatch tables), `WaveOpticsDock.__init__`
  (658 ln of widget construction), `MainWindow._create_docks` (375 ln). `_run_impl` in particular
  mixes: config parsing, global-state mutation, source construction, the lens router, the
  whole-prescription short-circuit (an entire duplicated finalisation block at `:762-820`
  including a second `results.update({...})` and a second `beam_d4sigma` import), the per-surface
  loop, the to-focus dispatch, the detector model, file IO, and `PropagationResult` assembly.
  Proposed: extract `_build_source`, `_run_lens_router`, `_run_surface_loop`, `_propagate_to_focus`,
  `_finalise(results)` — the last one removes the duplicated finalisation outright.
* **Duplicated dock scaffolding**: `minimumSizeHint`/`sizeHint` returning `QSize(40,40)`/`QSize(400,200)`
  is copy-pasted verbatim (docstring included) into at least 5 modules
  (`waveoptics_dock:2925`, `log_viewer_dock:394`, `element_table:1309`, `layout_2d:1091`,
  `layout_3d`). The `ax.set_facecolor('#0a0c10')` / `tick_params(colors='#7a94b8')` /
  `spines.set_color('#2a3548')` triple appears in ~20 docks. Both want a
  `ui/_dockbase.py::AnalysisDockMixin` + `style_axes(ax)` helper.
* **Worker boilerplate duplicated 16×** with three different cancellation conventions
  (`isInterruptionRequested`, `CancellableProgress.should_stop`, nothing at all) and two different
  finished-signal names. A single `ui/_worker.py::AnalysisWorker(QThread)` base — snapshot in
  `__init__`, `finished_result` signal, `check_interrupt()` helper, guaranteed-emit `run()` wrapper
  — would delete several hundred lines and fix the P1 shutdown bug structurally.
* **Dead / unreachable code** confirmed: `start_elem`/`end_elem` (P2), the image-plane-WFE block
  (P3), `_suppress_history` (P3), the identical `if/elif` in `to_prescription` (P0 note),
  `layout_2d:614`'s `* 0.0`, and `_run_impl:948` `if used_lens_router: … else: pass`.
* `ui/__init__.py`'s architecture comment is honest and up to date (it records the v5.30 deletion
  of the superseded spreadsheet editor and notes `workers.py` never existed).
* `run_lumenairy_designer.py` is clean; its comment about the old `lumenairy.prescriptions`
  import is accurate and the current `import lumenairy as la` path works. One nit: the
  `wv_nm = wv_nm * 1e9` guard at line 47 only fires `if isinstance(wv_nm, float)`, so an int
  wavelength in metres would be passed through as nanometres.
* `requirements-gui.txt` is consistent with the code (PySide6 ≥ 6.5, matplotlib, scipy ≥ 1.13,
  pyvista/pyvistaqt for `layout_3d`, h5py; pyfftw/numexpr/cupy correctly optional). Nothing in
  `ui/` imports an undeclared GUI dependency.

---

## Unverified suspicions

* `layout_2d._draw_rays` uses `surface_frames_2d_mm()` (one frame per surface + the Detector)
  while `run_trace` appends an image plane at the **paraxial BFL** when the Detector distance is 0.
  In that case the last drawn ray segment should terminate at the wrong place. I could not render
  it without Qt; the index arithmetic at `layout_2d.py:912-926` suggests it extrapolates 20 mm
  instead, which would be visibly wrong but not catastrophic.
* `_filter_wave_optics_surfaces(unfold_mirrors=False)` leaves Zemax-signed negative thicknesses in
  the list, and `_run_impl` guards the propagation with `if ts.thickness > 0:` — so with the
  checkbox off, every post-mirror gap is silently skipped (not propagated backwards, just dropped).
  I did not confirm the checkbox default is on.
* `main_window._create_docks` (375 lines) and `workspace.py` were only skimmed; the
  `DEFAULTS_REVISION = 7` migration path for saved workspaces was not exercised.
* The AO / phase-retrieval / coronagraph / coherence docks (≈3600 lines combined) were read only at
  skim level for threading and global state; their physics glue was not audited line by line.

---

## Checked and found correct

* **Coordinate breaks in `ui/model.py` are now correct and consistent with `raytrace/world.py`** —
  the `CONVENTIONS.md` §7 claim holds. `recompute_element_frames` (`model.py:1305-1331`) uses
  `R_math(+θ)` forms composed `R @ Rx @ Ry`, matching `world._rot_x/_rot_y` and
  `tilt_R = _rot_x @ _rot_y @ _rot_z` with `new_R = R @ tilt_R`. Decenter-then-tilt matches
  `order == 0`. Measured (`t3_cb.py`): a `+90°` `tilt_x` puts local +z at world `(0, −1, 0)` on
  both sides, and for a 30° tilt + (2, 1) mm decenter the UI's world frames match
  `world_surfaces_from_prescription(model.to_prescription())` **exactly** in orientation and in
  relative position (the only difference is the constant 10 mm leading air gap, which the
  prescription format cannot represent).
* **Singlet prescription export is byte-equivalent to `make_singlet`** — measured (`t1_singlet.py`):
  the exported `surfaces`/`thicknesses`/`aperture_diameter` match
  `la.make_singlet(R1=50e-3, R2=inf, d=3e-3, glass='N-BK7', aperture=25.4e-3)` key for key, and
  `system_abcd` gives identical EFL 99.288517 mm / BFL 97.293283 mm from both, matching
  `model.efl_mm` / `model.bfl_mm`.
* **`aperture_diameter` is a diameter and `semi_diameter` a radius, consistently** —
  `load_prescription:1760-1762` (`epd_mm = aperture*1e3`, `sd_default = aperture*1e3/2`),
  `to_prescription:2895` (`aperture_diameter: self.epd_m`), and the wave dock's aperture clip
  `E[R_sq > ts.semi_diameter**2] = 0` all agree.
* **Refraction phase-screen sign** — `waveoptics_dock.py:890` `phase = -k*(n2-n1)*sag` is the
  correct vertex-plane-referenced transmission for `exp(+ikz)` (OPL = `(n1-n2)·sag`), and matches
  `CONVENTIONS.md` §7's `phi = -k n_substrate * sag`.
* **`system_abcd` is not corrupted by the inserted coord-break Surfaces** — measured
  (`t10_abcd_cb.py`): a two-singlet system gives EFL 72.971442 / BFL 40.112087 mm with and without
  a cb in the local list (identical to 6 decimals); the cb is handled as a zero-thickness flat
  air-air surface.
* **Airy-radius and f/# formulas** — `analysis.py:263` (`1.22 λ|efl|/EPD`),
  `analysis.py:390` (`1.22 λ|efl|/(2·sd)`), `spot_field_dock.py:187` (`1.22 λ·efl/(2·semi_ap)`),
  `main_window._update_status:2831` (`|efl|/epd`, both mm), `element_table._update_info:1276`
  (`|efl|/epd_m`, both m), `model.merit_function:2539` (`|efl|/epd_m`, both m) are all
  dimensionally consistent and match `system_abcd`'s documented reduced-EFL convention
  (`seidel.py:220-230` explicitly blesses `ui/model.py`'s `na = epd/(2|efl|)` and
  `ui/spot_field_dock.py`'s Airy formula).
* **Principal-plane and ABCD unit conversions** — `analysis.py:270-295`: `H=(D−1)/C`,
  `H'=(1−A)/C` (Welford), `B·1e3` m→mm, `C/1e3` 1/m→1/mm. Correct.
* **No duplicated sag math** — `layout_2d.py:429` and `layout_3d.py:343` both call the library's
  `elements.lenses.surface_sag_biconic`; the wave dock does too (`:882-889`). No local conic/asphere
  re-implementation anywhere in `ui/`.
* **`lumenairy.ui` is not on the `import lumenairy` path** — measured: `import lumenairy` = 2.86 s,
  `ui in sys.modules: False`.
* **No matplotlib figure leaks** — every dock owns one `Figure()` and calls `self.fig.clear()` on
  redraw; `pyplot` appears only inside `repl_dock.py`'s user namespace (lines 44, 108), so no
  figures accumulate in pyplot's global registry.
* **Debouncing is in place** where it matters: source-form edits (200 ms,
  `element_table.py:462-465`), wavelength/EPD spinners (`:889-893`), auto-retrace (200 ms,
  `main_window.py:204-208`), autosave (1 s, `:189-193`), slider dock merit (`slider_dock.py:140-152`).
* **Worker→GUI signalling is correct** — all 16 workers communicate via Qt signals (auto/queued
  connection); no worker touches a widget directly. `WaveOpticsWorker` snapshots model state on the
  GUI thread in `__init__` and its `run()` wrapper guarantees exactly one `finished_result` emission
  on every path including a broken `__str__` (`:488-509`) — this is the pattern the other 15 should
  copy.
* **`_shutdown_dock_workers` exists and is wired** from `MainWindow.closeEvent` (`:1091-1103`); the
  design is right, only the per-worker cooperation is missing (P1 above).
* **`_stop` uses cooperative interruption, not `terminate()`** (`waveoptics_dock.py:2658-2673`) —
  the v5.17 P2-40 fix is real and correctly implemented on that dock.
* **`_WheelOnFocusFilter`** (`main_window.py:3458-3495`) is installed once via a guarded module
  singleton; idempotent, no leak.
* **`run_lumenairy_designer.py`** launches correctly (desk-checked): `apply_dark_theme`,
  `MainWindow()`, `--demo` via `la.thorlabs_lens`, `.zmx`/`.txt` via `la.load_zemax_zmx` /
  `la.load_zemax_prescription_data_txt`, all of which exist at the top-level namespace.

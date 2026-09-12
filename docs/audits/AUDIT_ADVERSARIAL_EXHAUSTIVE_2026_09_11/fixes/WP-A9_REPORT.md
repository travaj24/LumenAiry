# WP-A9 — Designer UI (`lumenairy/ui/`) + `run_lumenairy_designer.py`

Branch `audit-fixes-2026-09`.  Findings U1–U7 (report §6, partition report
`UI.md`).  PySide6 is not installed on this machine; every number below was
measured through the auditor's Qt stub
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub`), which is
also the harness for the new tests — nothing skips on PySide6's absence.

Every "before" number was measured by running the auditor's repro scripts on
HEAD **before** any edit; every "after" by re-running the same script.  The
model-level regression assertions were additionally replayed against HEAD's
`ui/model.py` exec'd in-process (`prefix_check.py`, §4) to prove they fail
pre-fix: **17 of 17 fail or error on HEAD.**

---

## 1. Summary table

| ID | status | files:lines | tests (`tests/unit/test_audit2609_a9_ui.py::`) | oracle | measured before → after |
|---|---|---|---|---|---|
| **U1** (P0) | **fixed** | `ui/model.py:2895-2960` (`to_prescription` legacy loop), `:3010-3070` (return dict + `object_distance_m` / `_detector_distance_m`), `:31-57` (`SurfaceRow.is_stop`), `:1810-1845` (`load_prescription` stop), `:2245,:2356` (both trace builders) | `test_u1_exported_folded_prescription_matches_the_layouts_own_abcd`, `test_u1_is_mirror_and_is_stop_are_emitted_per_surface`, `test_u1_coord_break_air_gap_survives_the_legacy_export`, `test_u1_singlet_export_stays_byte_equivalent_to_make_singlet` | the model's OWN `build_trace_surfaces()` list — what the 2-D layout, the spot diagram and `model.efl_mm` are drawn from; plus `make_singlet` for the unfolded control | folded EFL **0.15886162762977662 → 0.17877415918110384 m** (= layout exactly); BFL **0.15620131539451035 → 0.12780222366203894 m**; 45° fold total gap **0.003 → 0.043 m** (local list 0.043); singlet control unchanged at EFL 0.09928851726861038 / BFL 0.09729328309216069 |
| **U2** (P0) | **partially fixed** (UI side complete; one library line outstanding — §5.1) | `ui/model.py:2920-2935` (`semi_diameter` / `is_mirror` / `is_stop` per surface) | `test_u2_every_surface_carries_its_own_semi_diameter`, `test_u2_is_stop_round_trips_through_load_prescription` | the `SurfaceRow.semi_diameter` the user typed | emitted per-surface sd **absent → [0.025, 0.006, 0.006] m**; `surfaces_from_prescription` trailing surface **0.0127 → 0.006 m**; the MIRROR is still clamped to 0.006 by `trace.py`'s `min()` against a mis-indexed `elements` match |
| **U3** (P1) | **fixed** | `ui/model.py:2470-2500` (`run_trace` point-source bundle), `:2990-3010` (`object_distance_m`) | `test_u3_point_source_bundle_is_a_real_cone`, `test_u3_object_distance_comes_from_the_element_geometry` | closed form `rho_k = (k/num_rings)·semi_ap/obj_dist` at azimuth `2πj/rays_per_ring` | unique directions **4 of 25 → 25 of 25**; `L == M` everywhere **True → False**; marginal `|rho|` **0.0898026 → 0.0635000** (exact: `semi_ap/obj_dist` = 0.0635) |
| **U4** (P1) | **fixed** | `ui/waveoptics_dock.py:412-460` (`_process_overrides`), `:692-703` (`run`), `:756` (removed), `:1922-1945` + `:2694-2710` ("Library default") | `test_u4_fft_backend_and_ram_cap_are_restored_after_a_run`, `test_u4_overrides_are_restored_when_the_run_raises` | `fft_infra.USE_PYFFTW` / `USE_SCIPY_FFT` / `memory.get_max_ram()` read before and after | a default Run left `USE_PYFFTW=False, USE_SCIPY_FFT=False` **permanently → restored on every exit path including an exception**; combo index 0 now touches nothing |
| **U5** (P1) | **fixed** | `ui/waveoptics_dock.py:899-1012` (router + `lens_model_used`), `:452-500` (`_prescription_from_surfaces`), `:385-404` (`_filter_wave_optics_surfaces` gap carry), `:3020-3035` (summary) | `test_u5_router_runs_the_unfolded_equivalent_instead_of_downgrading`, `test_u5_unfolding_preserves_the_axial_path`, `test_u5_folded_prescription_is_detected_both_ways` | `apply_real_lens`'s own refusal, and the total unsigned axial path of the surface list | folded design: **silent downgrade to the thin-screen ASM loop, labelled with the requested model → the requested model actually runs on the unfolded equivalent, summary says `(unfolded equivalent)`**; unfolding lost the mirror's 30 mm gap → total axial path conserved (0.071 m both ways) |
| **U6a** | **fixed** | `ui/waveoptics_dock.py:2946-2960` (`_on_finished`), `:1106-1112` + `:1461-1467` (`N_out`/`dx_out`) | `test_u6a_finished_slices_the_output_grid_not_the_input_grid` | `I_focus.shape[0]` | MFT-shaped result (`N`=256, `I_focus` 64²): **IndexError on the cross-section → 16-point slice on the 64-grid at `dx_out`** |
| **U6b** | **fixed** | `ui/main_window.py:3232-3243` | `test_u6b_insert_source_preset_uses_the_setter` | the property has no setter | all six Insert ▸ Source presets **AttributeError → set the source, checkpoint, sync λ, emit** |
| **U6c** | **fixed** | `ui/element_table.py:680-711` | `test_u6c_source_form_edit_preserves_wavelength_and_polarization` | the live source before the edit | after a form edit: λ **1310.0 → 632.8 nm** (kept), polarization **None → 'rcp'** (kept), `emitter_nx` **float → int** |
| **U6d** | **fixed** | `ui/model.py:880-910` (`set_wavelength` + `sync_source_wavelength`), `:838` (`set_source`), `:1758` (`load_prescription`) | `test_u6d_model_wavelength_is_carried_onto_the_source` | the same point source built at the model wavelength | max \|phase difference\| **0.21638168909173322 → 3.53e-17 rad** |
| **U6e** | **fixed** | `ui/model.py:59-77` (`_as_count`), `:89-96`; `ui/layout_2d.py:607,704`; `ui/layout_3d.py:558` | `test_u6e_emitter_counts_are_integers` | `to_source()` actually returning a field | `to_source` **TypeError (swallowed into a plane wave) → (64, 64) field**; `SourceDefinition` now rejects 0 / 3.5 with the §2 prefix |
| **U6f** | **fixed** | `ui/model.py:2370-2400` (`_build_trace_surfaces_world` gap + `_next_optical_element`) | `test_u6f_world_surface_list_carries_inter_element_air_gaps` | `find_paraxial_focus` / `system_abcd` on the LOCAL list (same physical system) | `find_paraxial_focus(world)` **0.05834391476238654 → 0.04011208679014488 m** (local value exactly, +45 % error removed); `system_abcd` EFL **0.061576549 → 0.072971442 m** |
| **U6g** | **fixed** | `ui/psf_mtf_dock.py:223-250` + `:322-360` (`_exit_pupil_rays`) | `test_u6g_psf_pupil_is_taken_at_the_exit_pupil_not_the_image_plane` | the marginal ray height = EPD/2; `rays.z` on one plane | attribute **`.opl` (does not exist) → `.opd`**; pupil radius **2.501e-04 m (image plane, 31 of 65536 cells) → 1.270e-02 m** (= EPD/2 to 2 %) |
| **U6h** | **fixed** | `ui/optimizer_dock.py:71-133` (`_detached_copy`), `:1520-1560` (`GlobalSearchWorker`), `:905-918` (`_on_finished`) | `test_u6h_optimizer_worker_never_touches_the_live_model` | the live element objects' identity and `origin` after N merit probes | after 3 probes the live `SurfaceRow.radius` and `Element.origin` **moved → bit-unchanged**; the clone moved as expected |
| **U6i** | **fixed** | `ui/_worker.py` (new), + entry / loop polls in `ao_dock`, `caustic_dock`, `coherence_dock` (×2), `coronagraph_dock`, `ghost_dock`, `multiconfig_dock`, `optimizer_dock` (×2), `phase_retrieval_dock`, `psf_mtf_dock`, `richards_wolf_dock`, `tolerance_dock`, `wavefront_map_dock`, `waveoptics_dock` (`_CalibrationWorker`) | `test_u6i_workers_honour_request_interruption`, `test_u6i_every_qthread_worker_exposes_a_cancellation_path` | `QThread.requestInterruption()` → `CancellableProgress.should_stop`; structural sweep over every `QThread` subclass | workers with a cancellation path **2 of 16 → 18 of 18** |
| **U7** | **fixed** (13 items) | see §2.7 | 9 `test_u7_*` + 1 `test_u6_no_worker_shadows_the_builtin_finished_signal` | per item | see §2.7 |

Repro scripts re-run after the fix: `t1_singlet`, `t2_mirror`, `t3_cb`,
`t4_pointsrc`, `t5_wv`, `t6_srcprop`, `t7_emitter`, `t8_docks`, `t10_abcd_cb`,
`t11`.  `t3_cb`'s coordinate-break verification is unchanged (the UI world
frames still match `world_surfaces_from_prescription` exactly in orientation and
relative position).  `t9_psfpupil` re-implements the dock's binning inside the
script rather than calling it, so its printed "31 of 65536" is unchanged by
design; the dock path itself is measured by `test_u6g_*`.

---

## 2. Per finding

### 2.1 U1 — `to_prescription()` stripped `is_mirror` and leaked Zemax-signed negative thicknesses  [P0]

**Wrong.**  The legacy `surfaces` loop emitted eight geometry keys and nothing
else.  `build_trace_surfaces()` marks mirrors `is_mirror=True` and carries
Zemax-signed (negative) post-mirror thicknesses — a self-consistent pair,
because `system_abcd` flips the index sign at a mirror (`n2 = -n1`) so
`t/n_after = (-0.03)/(-1) = +0.03`.  Dropping only the flag left the negative
thickness against a POSITIVE index: a literal backwards propagation, plus a
mirror modelled as an air→air no-op.  Coord-break Surfaces were `continue`d and
their transfer thickness — real axial distance in the post-cb frame — simply
lost; the `elif` at `model.py:2779-2782` was dead (both branches identical).

**Changed.**  The loop now emits `semi_diameter`, `is_mirror` and `is_stop` per
surface, accumulates a skipped coord-break's thickness into the gap that follows
the previously emitted surface (`carry`), and the dead `elif` is gone.
`thicknesses` keeps its `len(surfaces) - 1` length, which
`validate_prescription` accepts alongside `len(surfaces)`.  The returned dict
also gained `object_distance`, `image_distance` and `stop_index`.

`is_stop` required a home in the model: `SurfaceRow.is_stop` (new), read by
`load_prescription` from either a per-surface `is_stop` or a `stop_index`,
carried by both trace-surface builders into `Surface(is_stop=…)`, and persisted
in the session JSON.  Previously the .zmx `STOP` keyword could not survive an
import at all.

**I did NOT refuse the legacy keys for folded systems** (the audit's alternative
fix).  Emitting `is_mirror` makes them *correct* — the exported ABCD now
reproduces the layout's bit-for-bit — so refusing them would break 18 call sites
for no gain.  `allow_unfolded_equivalent` is handled where it belongs, in the
lens-model router (U5).

**Verified.**
* Exported vs layout ABCD on the audit's fold fixture: EFL
  `0.17877415918110384` vs `0.17877415918110384`, BFL `0.12780222366203894` vs
  `0.12780222366203894` — identical to the last bit.  Pre-fix: 0.15886162762977662
  / 0.15620131539451035 (11.1 % / 22.2 % apart).
* 45° fold (`t11.py`): `LOST (the cb_post air gap): 0.04 → 0.0` m.
* Unfolded singlet control (`t1_singlet.py`): still key-for-key identical to
  `make_singlet`, EFL 99.288517 mm / BFL 97.293283 mm from both.
* `.zmx` export → `load_zemax_zmx` → `load_prescription` of a doubly-tilted fold
  still round-trips: element types `['Source','Mirror','Singlet','Detector']`,
  tilts `[0, 45, 45, 0]`.
* `user_library._serialize_prescription` round-trips the new keys (the `inf`
  semi-diameter behaves exactly like the existing `inf` radius).

**Residual risk.**  Consumers that read `surfaces`/`thicknesses` and ignore
`is_mirror` see exactly what they saw before (same values, one extra key), so
they cannot regress; consumers that honour it get correct physics.  A
prescription saved to the user's lens library BEFORE this fix still carries the
old shape and will still be mis-analysed when re-loaded — it has no mirror
markers to recover.  Noted as a deferred item (§6.1).

### 2.2 U2 — per-surface semi-diameters re-indexed onto the wrong surfaces  [P0]

**Wrong.**  With `semi_diameter` absent, `surfaces_from_prescription` fell back
to `refr_elems = [e for e in elements if e['element_type'] == 'surface']`
indexed by `i` over the FULL surface list.  Every mirror shifts the mapping by
one, and `i >= len(refr_elems)` leaves the trailing surface at
`aperture_diameter / 2`.

**Changed.**  Every legacy surface now carries its own `semi_diameter`.

**Verified, and the limit of the UI-side fix.**  `trace.py:542-552` honours the
per-surface key and then takes `min(sd, elem_sd)` with the broken index, so on
the audit fixture (mirror sd 25 mm, lens sd 6 mm):

| surface | before | after | correct |
|---|---|---|---|
| mirror R = −200 | 0.006 | 0.006 | 0.025 |
| lens S1 | 0.006 | 0.006 | 0.006 |
| lens S2 | **0.0127** | **0.006** | 0.006 |

Every refracting surface is now exact.  The mirror remains clamped to the
following lens's semi-diameter because the `min()` cannot be defeated from the
producing side — the exact one-line `trace.py` change is in §5.1.  Status:
**partially fixed**, and the test asserts what my side controls (the emitted
dict, exactly) plus the now-correct refracting surfaces, so it does not pin the
residual defect.

### 2.3 U3 — degenerate point-source bundle  [P1]

**Wrong.**  `for t in theta:` never used `t`; both direction cosines got the
full `frac * semi_ap / obj_dist`.  `num_rings × rays_per_ring` rays became
`num_rings` distinct directions on the x = y line, each duplicated
`rays_per_ring` times, and `|rho| = sqrt(2) * (semi_ap/obj_dist)` over-filled the
pupil by 41 %.  Secondarily, `obj_dist` came from
`SourceDefinition.object_distance_mm` while rays launch at world z = 0 and the
first surface sits at `elements[1].distance_mm`.

**Changed.**  `rho = (ring/num_rings) * semi_ap / obj_dist`, `L = rho·cos t`,
`M = rho·sin t`; `obj_dist` from the new `SystemModel.object_distance_m()` (the
geometric source-to-first-optic gap), falling back to the form field only when
no optic is placed.

**Verified.**  `t4_pointsrc.py` at `num_rings=3, rays_per_ring=8`: unique (L, M)
pairs **4 → 25**, `L == M everywhere: True → False`, ring radii exactly
`[1/3, 2/3, 1] × 0.0635` and azimuths uniform to 1e-12, chief ray on axis.

**Residual risk / behaviour change.**  A design whose form field disagreed with
the element placement now fills the pupil differently.  Migration note is in the
changelog; the form field's tooltip and `GUI_README.md`'s source table now say
the field is advisory.

### 2.4 U4 — a default run disabled pyFFTW and SciPy FFT process-wide  [P1]

**Wrong.**  Unconditional `USE_PYFFTW = USE_SCIPY_FFT = False` at the top of
`_run_impl`, from the worker thread, never restored; `set_max_ram` likewise.
Combo index 0 was `NumPy FFT`, so the *default* Run triggered it.

**Changed.**  `_process_overrides(backend, mem_limit_gb)` — a context manager
wrapping the whole run in `WaveOpticsWorker.run()`, saving and restoring both
FFT flags and the RAM cap on every exit path.  New combo index 0 **"Library
default"** maps to `backend='default'`, which touches nothing.

**Verified.**  Flags and `get_max_ram()` read before/after: unchanged for
`'default'`, `'numpy'`, `'scipy'` and for a run that raises.  `tests/conftest.py`
names this site as the reason `fft_infra` is in its leak-guard list; the leak it
describes is gone (§5.3).

### 2.5 U5 — the lens-model router downgraded silently  [P1]

**Wrong.**  `except Exception: … used_lens_router = False` dropped through to the
per-surface ASM loop with no report.  `apply_real_lens` refuses a folded
prescription **by design**, so every fold silently produced a thin-screen PSF
labelled with the analytic/traced/Maslov model the user chose.

**Changed.**  `results['lens_model_requested' / 'lens_model_used' /
'lens_model_fallback_reason']` and a summary line.  More importantly, with
"Unfold mirrors" ticked the router now hands the core function the
unfolded-equivalent prescription that the per-surface loop would have walked
(`_prescription_from_surfaces(trace_surfs, …)`), so the requested model really
runs.

**A complication I hit and resolved.**  Emitting `is_mirror` (U1) makes the
prescription trip a SECOND, stricter guard in `_lens_real.py:5375` — a
per-surface mirror check that, unlike `_check_no_silent_fold_drop`
(`:2503`), does **not** honour `allow_unfolded_equivalent`.  So the handshake
the audit suggested no longer suffices.  Handing the library an already-unfolded,
mirror-free prescription is strictly better anyway: the user gets the model they
asked for on the approximation they ticked, and the library is not asked to
pretend a mirror is a refractor.  The two-guard inconsistency is reported in
§5.2.

**Also fixed here.**  `_filter_wave_optics_surfaces(unfold_mirrors=True)` dropped
a mirror Surface *and its thickness*, shortening the unfolded path by the whole
mirror-to-next-element gap (30 mm on the audit fixture).  A dropped surface's
thickness is now carried onto the previous kept surface; total unsigned axial
path is conserved (pinned).

### 2.6 U6 — the crash / λ-mismatch list

Eight items; all fixed.  See the summary table for the measured numbers and the
changelog for the user-facing description.  The two with design judgement in
them:

* **U6h (optimizer data race).**  The class comment claimed the worker "never
  mutates the shared live model off the GUI thread"; `apply_result=False` only
  governed the final write-back, while `merit_function` → `set_variable_values`
  → `_invalidate` → `recompute_element_frames` rewrote every element's `origin` /
  `R` and nulled `_flat_surfaces_cache` on every scipy probe.  Both
  `OptimizeWorker` and `GlobalSearchWorker` now build a detached `SystemModel`
  deep copy in `__init__` (on the GUI thread) and hand back only `result_x`.
  `GlobalSearchWorker` used to apply its best-so-far directly from the thread;
  it now sets `apply_result_on_failure = True` so the dock still applies the
  best design on cancel, on the GUI thread.
* **U6i (interruption).**  The audit's recommendation was to "make
  `CancellableProgress` read from `requestInterruption()`", but `progress.py` is
  outside my ownership.  I implemented the same semantics on my side of the
  boundary: `ui/_worker.py::ThreadCancellableProgress` subclasses it and ORs in
  the thread's flag, so every existing `self._cancel_progress.should_stop` poll
  now honours both channels — and, because it is still a valid `progress=`
  callback, any library call the worker passes it to becomes interruptible at the
  library's own checkpoints.  The workers with no progress object got an entry
  poll; `coronagraph_dock` got one at each of its four stage boundaries and
  `ao_dock`'s `_stop_requested` became a property reading both channels.

### 2.7 U7 — the P2 / P3 list

| item | file | status | measured |
|---|---|---|---|
| workers shadow `QThread.finished` | `optimizer_dock.py:75,1523`, `tolerance_dock.py:42` | fixed (renamed `finished_result`) | sweep over all 18 worker classes: shadowing **3 → 0** |
| dead "Start at / End at" | `waveoptics_dock.py:768-793` + `model.py:2050-2070` (`element_surface_spans`) | fixed (slices `trace_surfs`, re-derives its own BFL, reports the range, declines the whole-prescription router) | both locals were assigned and never referenced |
| `power_in` at the wrong plane | `waveoptics_dock.py:879-886` | fixed (captured from `E` at construction) | unticking "Source" made Throughput ~100 % |
| `build_run_trace_world_surfaces(image_distance=)` ignored | `model.py:2104-2126` | fixed (explicit argument wins; `None` keeps the Detector) | image plane at the detector 100 mm vs the passed BFL 97.293 mm |
| `get`/`set_variable_values` disagree | `model.py:2735-2790`, `delete_element`/`move_element` re-base | fixed (`live_opt_variables()`, sized `ValueError`) | a stale triple made the setter write into the wrong parameter |
| blocking ASM benchmark on the GUI thread | `waveoptics_dock.py:49-77`, `:405-425` (`_CalibrationWorker`), `:2080-2124` | fixed (background thread; 12 ms fallback meanwhile) | three 512² FFT pairs inside a handler wired to 14 widgets |
| `File ▸ New` re-runs `__init__` | `main_window.py:2184-2196` + `model.py:1540-1575` (`reset_design`) | fixed | `prefs['ray_color']`, `theme`, `lens_options`, `auto_retrace_mode` survive; elements/λ/EPD/variables reset |
| `_apply_real_lens_asm_equiv` calibration | `waveoptics_dock.py:285-302` | fixed (`1.1·(n−1)`, the exact fit) | n=2: **1.4 → 1.1**; n=3: **2.6 → 2.2** (docstring's own points) |
| `set_display_distance` duplicates the helper | `model.py:806-818` + `element_z_positions_mm` | fixed (both read the cached frames) | absolute-Z round trip: distance **40.0 → 37.0 mm** per write-back → unchanged |
| `min_thickness` merit offset | `model.py:2825-2840` | fixed (`surfaces[:-1]`) | compliant design merit **1.0 → 0.0**; a real 0.25 mm violation still scores `(1−0.25)²` |
| merit label always µm | `model.py:2930-2940` | fixed (unit only for `rms_spot`) | — |
| dead `_suppress_history` | `model.py:536`, `:1505`, `bulk_edit()` | fixed (working context manager) | a 3-mutator block is **3 → 1** undo step |
| `layout_2d` `* 0.0` | `layout_2d.py:604-618` | fixed (side view draws y rows only) | `nx` identical overlapping dots removed |
| worker snapshot aliases the model's Surfaces | `waveoptics_dock.py:614-628` | fixed (deep copy) | the class contract now holds |
| `analysis.py` image-plane WFE unreachable | `model.py` emits `object_distance` | fixed | key **absent → 0.1 m** for a point source, **0.0** for a plane wave |
| session drops polarization / fiber fields | `model.py:1730-1760` | fixed | polarization **None → 'linear_45'**, MFD **10.4 → 9.2 µm** after a round trip |
| `diagnostics` hard-codes `\` | `diagnostics.py:79-82` | fixed (`os.path.basename`) | — |
| matplotlib at module scope | `ui/_mpl.py` (new) + 15 modules | fixed | importing 15 dock modules leaves `matplotlib.figure` out of `sys.modules` |
| a worker base class | `ui/_worker.py` (new) | added | `AnalysisWorker` + `ThreadCancellableProgress` + `interrupt_check` |
| `run_lumenairy_designer` int wavelength | `run_lumenairy_designer.py:49-58` | fixed | an int metre value is now converted |

---

## 3. Files touched

**Modified (26).**  `lumenairy/ui/`: `model.py`, `waveoptics_dock.py`,
`optimizer_dock.py`, `psf_mtf_dock.py`, `main_window.py`, `element_table.py`,
`layout_2d.py`, `layout_3d.py`, `diagnostics.py`, `tolerance_dock.py`,
`multiconfig_dock.py`, `phase_retrieval_dock.py`, `coronagraph_dock.py`,
`ao_dock.py`, `caustic_dock.py`, `coherence_dock.py`, `ghost_dock.py`,
`richards_wolf_dock.py`, `wavefront_map_dock.py`, `distortion_dock.py`,
`footprint_dock.py`, `glass_map_dock.py`, `jones_pupil_dock.py`,
`lg_aberration_dock.py`, `rayfan_dock.py`, `shack_hartmann_dock.py`,
`spot_field_dock.py`, `thin_grating_dock.py`; plus `run_lumenairy_designer.py`
and `GUI_README.md` (two lines: the source table's object-distance note and the
wave-optics backend list).

**New (3).**  `lumenairy/ui/_mpl.py`, `lumenairy/ui/_worker.py`,
`tests/unit/test_audit2609_a9_ui.py`.

Nothing outside `ui/` + `run_lumenairy_designer.py` + `GUI_README.md` + my own
test file was edited (`git diff --name-only` confirms; the other modified paths
in the working tree belong to the concurrently-running WPs).

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`.

| command | result | duration |
|---|---|---|
| `python -m pytest tests/unit/test_audit2609_a9_ui.py -q -p no:cacheprovider` | **36 passed** | 5.5 s |
| the 13 existing UI test files together with the new one (`test_audit_p1_gui_dead_import`, `test_audit_s4_2_optimizer_bounds_units`, `test_audit_s4_3_waveoptics_biconic`, `test_audit_s4_6_wave_optimizer_thickness`, `test_audit_s4_7_optimizer_thread_and_bounds`, `test_audit_w5_ui`, `test_audit_w6_ui`, `test_g08_s4_20_packaging_ui`, `test_niche_audit_w3_ui_deprecation`, `test_v5_4_2_run_trace_empty_prescription`, `test_v5_4_6_io_ui_delegated`, `test_v5_14_5_viewer_polarization`) | **83 passed, 38 skipped** | 24 s |
| the same 12 files WITHOUT my test file (control) | 47 passed, 38 skipped | 9.5 s |
| `test_audit_lens`, `test_niche_d4_dgrating`, `test_v4_15_agent_e`, `test_v4_15_1_agent_e`, `test_v5_4_phase_masks`, `test_v5_4_chebyshev_fit_2d`, `test_v5_4_zernike_normalization_weighting`, `test_niche_audit_w3_oracles` | 343 passed, 24 skipped, **4 failed** (see below) | 301 s |
| `python validation/run_all.py layout_shrink` | 1 file passed (skips internally, no GUI deps — unchanged) | 0.1 s |
| the 10 auditor repro scripts under `repro/UI/` | re-run before and after; numbers in §1 | — |
| `prefix_check.py` (HEAD's `ui/model.py` exec'd in-process) | **17 of 17 assertions fail or error pre-fix** | 3 s |

**Skip-set is unchanged.**  The 38 skips are exactly the ones the same files
produce without my test file.  That took work: the Qt stub is installed at my
module's import, used to import the 20 `lumenairy.ui` modules the tests need,
and then *parked* out of `sys.modules` — pytest imports every test module during
collection, and the sibling UI files guard themselves with
`try: import PySide6 / except ImportError: skip`, so a live stub pulled them out
of their skip and they failed (18 failed / 6 errors) against a Qt that cannot
paint.  An autouse fixture un-parks the stub for the duration of each of my
tests only.

**Pre-existing failures, unrelated to WP-A9** (4, all in
`tests/unit/test_niche_audit_w3_oracles.py`):
`test_w3_t3b_lg_merit_responds_to_a_curvature_change`,
`test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged`,
`test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`,
`test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged`.  They exercise
`lumenairy.LGAberrationMerit` → `propagators/asymptotic*.py` and
`asymptotic_modes.decompose_lg`; the failing numbers are LG-mode / σ-grid pins
(`1.31728430779321` against a pinned `8.8334897780e-14`; a complex sigma-grid
value off by 6e6 relative).  Those five `asymptotic*` modules are modified in the
working tree by the concurrently-running asymptotic WP; nothing in
`lumenairy/ui/*` is on their import path (my own
`test_u7_ui_package_is_not_on_the_import_lumenairy_path` proves `import
lumenairy` does not pull `lumenairy.ui`).  **My judgement: caused by the
asymptotic WP's in-flight edits, not by WP-A9.**  Flagged to the orchestrator.

**How the new tests were shown to fail pre-fix.**  `prefix_check.py` (in the
session scratchpad, reproduced in §7) fetches `git show HEAD:lumenairy/ui/model.py`,
execs it as a private module with `__package__ = 'lumenairy.ui'` so its relative
imports resolve, and replays the 17 model-level assertions against THAT class.
All 17 fail or error, printing the pre-fix values: exported EFL
`0.15886162762977662`, `semi_diameter` keys `[None, None, None]`,
`sd_from_prescription [0.006, 0.006, 0.0127]`, `unique dirs 4 of 25`,
`rho_max 0.089803`, `source λ 1310.0` against model 632.8, `emitter_nx type
float`, `focus world 0.05834391476238654` vs local `0.04011208679014488`,
`min_thickness merit 1.0`, `object_distance ABSENT`, absolute-Z `40.0 → 37.0`,
`restored polarization None`.  The dock-level items (U4, U5, U6a–c, U6g–i, U7)
are pinned by assertions on code that did not exist pre-fix (the context
manager, the helpers, the renamed signals) or, for `test_u6a` / `test_u6g` /
`test_u6h`, by behaviour that pre-fix raised (`IndexError`, `AttributeError`) —
which the audit measured directly.

---

## 5. Requested changes outside my ownership

### 5.1 `raytrace/trace.py` — finish U2 (one line, plus an index fix)

`surfaces_from_prescription`, lines 542-552 (WP-A1 / raytrace owner):

```python
        ps_sd = ps.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        # If elements list has per-surface semi-diameters, use the tighter one
        if elements is not None:
            refr_elems = [e for e in elements if e.get('element_type') == 'surface']
            if i < len(refr_elems):
                elem_sd = refr_elems[i].get('semi_diameter', np.inf)
                if elem_sd > 0 and np.isfinite(elem_sd):
                    sd = min(sd, elem_sd)
```

Two independent defects:

1. **The `elements` matcher runs even when the per-surface key is present**, and
   `min()`s against it.  Making it an `elif` is the minimal fix and is what the
   audit's own fix text assumes ("then the per-surface key wins … and the
   fragile `elements` matcher is bypassed"):

   ```python
        ps_sd = ps.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        elif elements is not None:
            ...
   ```

2. **The index is wrong whenever a mirror is present**, for any prescription
   that does not carry the per-surface key.  `refr_elems` excludes mirrors while
   `i` counts them.  The UI's `elements` list is one entry per surface in the
   same order as `surfaces`, so the correct match is positional over the whole
   list with a type check:

   ```python
        elif elements is not None and i < len(elements):
            elem_sd = elements[i].get('semi_diameter', np.inf)
            if elem_sd > 0 and np.isfinite(elem_sd):
                sd = min(sd, elem_sd)
   ```

   (Any loader whose `elements` list is NOT index-aligned with `surfaces` would
   need `surf_num` matching instead; all four loaders I checked are aligned.)

With (1) alone, the UI side is complete and U2 closes.  With (2) as well, every
other producer of a mirrored prescription is fixed too.  Measured effect on the
audit fixture: mirror semi-diameter `0.006 → 0.025` m.

### 5.2 `elements/_lens_real.py` — two mirror guards disagree about the opt-out

`_check_no_silent_fold_drop` (`:2491-2513`) offers
`prescription['allow_unfolded_equivalent'] = True` as the documented escape.
The per-surface guard added by audit P1-A (`:5363-5390`) raises on any
`surfaces[i]['is_mirror']` **unconditionally** and its message does not mention
the flag.  So a caller that sets the flag as the first guard instructs still
hits the second.  Requested: either honour `allow_unfolded_equivalent` in the
per-surface guard too, or drop the flag from the first guard's message so the
two agree.  Not urgent for the UI — the dock now unfolds before calling — but it
is a documented-option-that-does-not-work of exactly the kind §15.3 catalogues.
Owner: WP-A2 / WP-A3.

### 5.3 `tests/conftest.py` — a stale comment (no code change needed)

Lines 245-255 explain that `lumenairy.propagators.fft_infra` is in
`_LEAK_GUARD_MODULES` because "`lumenairy/ui/waveoptics_dock.py` clears
`USE_PYFFTW` unconditionally".  It no longer does (U4).  The guard should stay —
it is a good cross-module invariant — but the sentence should be updated to say
the UI leak is fixed and the guard now protects against recurrence.  I did not
edit a shared conftest.

### 5.4 `lumenairy/progress.py` — optional consolidation

`ui/_worker.py::ThreadCancellableProgress` exists only because
`CancellableProgress.should_stop` cannot see Qt's interruption flag.  If the
library ever wants to own this, `CancellableProgress` could grow an optional
`extra_stop_predicate` callable and the UI subclass would collapse to a one-line
construction.  Low value; noted for completeness.

---

## 6. Deferred

### 6.1 Migration for lens-library entries saved before this fix  [small]

A folded prescription saved to the user library by the pre-fix
`to_prescription()` has no `is_mirror` and no per-surface `semi_diameter`, so
re-loading it still analyses an air→air no-op.  The `elements` list in those
files DOES carry `element_type: 'mirror'` correctly, so a one-pass upgrade is
possible: in `user_library.load_lens`, when `surfaces` lacks `is_mirror` but
`elements` has mirror entries, back-fill `is_mirror` / `semi_diameter` /
`is_stop` onto `surfaces` by position.  ~30 lines in `user_library.py` (not my
file) plus a test.  Effort: ~1 h.  Not attempted here because the file is
outside my ownership and the affected artefacts are user data, not repo state.

### 6.2 `WaveOpticsWorker._run_impl` decomposition  [medium]

Still 685 lines with the duplicated finalisation block at `:762-820` (a second
`results.update({...})` and a second `beam_d4sigma` import).  I added `N_out` /
`dx_out` / the lens-model keys to BOTH copies, which is exactly the maintenance
tax the audit describes.  Design: extract `_build_source(cfg, snap)`,
`_run_lens_router(E, …) -> (E, used, meta)`, `_run_surface_loop(E, trace_surfs,
…)`, `_propagate_to_focus(E, …)` and `_finalise(results, …)`; the last removes
the duplicated block outright.  Effort: ~4 h plus a careful before/after
bit-identity check on three methods × two grids.  Deferred as too large for a
correctness pass.

### 6.3 A real aperture-stop UI  [medium]

`SurfaceRow.is_stop` now exists, round-trips and is exported, but nothing in the
UI can SET it — there is no Stop column in the surface sub-editor and no
"Make stop" action.  A design imported from a .zmx keeps its stop; a design
built in the GUI has none.  Design: one checkbox column in
`SurfaceSubModel.COLUMNS` with a radio-like exclusivity rule (setting one clears
the others) plus a context-menu action.  Effort: ~2 h.  Out of scope for the
findings as written.

### 6.4 The remaining `AnalysisWorker` migration  [medium]

`ui/_worker.py::AnalysisWorker` is in place and tested, but the 16 existing
workers were fixed in situ (cancellation path + signal rename) rather than
re-based on it, because re-parenting each one touches its dock's signal wiring
and payload shapes.  Design: migrate one dock per commit, starting with the four
that already have a `work()`-shaped body (`caustic`, `ghost`,
`richards_wolf`, `wavefront_map`).  Effort: ~30 min each.

### 6.5 Lazy dock construction  [small–medium]

`main_window.py:20-37` still imports 14 dock modules eagerly and `_create_docks`
constructs all of them at startup.  With matplotlib now lazy the remaining cost
is the dock widgets themselves.  Design: move the imports into `_create_docks`
and build non-default docks on first show.  Effort: ~2 h, needs a real PySide6 to
validate.  Not attempted headless.

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A9_CHANGELOG.md`
— two clearly separated blocks, one for `CHANGELOG.md` (library-facing effects:
the prescription contract, the FFT-flag leak, the import cost, the
object-distance convention change with its migration note) and one for
`GUI_CHANGELOG.md` (the full U1–U7 user-facing list).

The pre-fix verification script is at
`…/scratchpad/wp_a9/prefix_check.py` (session scratchpad, not committed).
